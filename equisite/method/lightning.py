"""
binding_task_lightning_setpred.py  —  Lightning module for UniSite set-prediction on GDEGAN.

NEW FILE (non-destructive). Wraps the FROZEN encoder (models/gda_hegnn.py -> (h, X)) with the
self-nominated-center head (models/setpred_center_head.py).

Two modes (head.use_mask_head):
  * False (exp-1): masks = foreground(p) split by nearest center. No per-site mask supervision.
  * True  (exp-2): each center paints its OWN mask (SetPredMaskHead), supervised PER-SITE with
    Dice+BCE — the evidence-based fix for the exp-1 mask bottleneck.

Eval: residue-mask IoU-AP via utils/mean_ap.eval_map at BOTH IoU 0.5 and 0.3 (+ centroid DCC).
"""


from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from equisite.backbone.gda_hegnn import GDEGAN as Encoder              # EquiSite's own copy of the encoder
from equisite.method.center_head import (
    SetPredCenterHead, SetPredCenterLoss, decode_setpred,
    SetPredMaskHead, decode_setpred_mask, batched_dice_loss,
)
from equisite.utils.losses import FocalLoss2
from equisite.utils.mean_ap import eval_map
from equisite.utils.utils import CosineCutoff


@dataclass
class SetPredResults:
    ap50: float = 0.0
    ap30: float = 0.0
    dcc_mean: float = 0.0
    dcc_success: float = 0.0
    num_pred: float = 0.0
    num_gt: float = 0.0


class SetPredLightningModule(pl.LightningModule):
    def __init__(self, model_config: dict, optimizer_config: dict,
                 loss_config: dict, head_config: dict, eval_config: dict = None):
        super().__init__()
        self.save_hyperparameters()

        mp = dict(model_config)
        cf = mp.pop("cutoff_fn", {}) or {}
        mp["cutoff_fn"] = CosineCutoff(cutoff=cf.get("cutoff", 10.0))
        self.model = Encoder(**mp)

        hd = model_config.get("n_atom_basis", 128)

        # split head_config into center-head kwargs and mask-head/loss knobs
        hc = dict(head_config or {})
        self.mask_mode = hc.pop("use_mask_head", False)
        mask_n_rbf = hc.pop("mask_n_rbf", 16)
        mask_cutoff = hc.pop("mask_cutoff", 20.0)
        self.w_mask_dice = hc.pop("w_mask_dice", 2.0)
        self.w_mask_bce = hc.pop("w_mask_bce", 1.0)
        mask_focal_alpha = hc.pop("mask_focal_alpha", 0.25)
        mask_focal_gamma = hc.pop("mask_focal_gamma", 2.0)
        self.mask_dice_smooth = hc.pop("mask_dice_smooth", 1.0)
        use_equiv_mask = hc.pop("use_equiv_mask", True)
        self.head = SetPredCenterHead(hidden_dim=hd, **hc)
        self.criterion = SetPredCenterLoss(**(loss_config or {}))

        if self.mask_mode:
            self.mask_head = SetPredMaskHead(hidden_dim=hd, n_rbf=mask_n_rbf, cutoff=mask_cutoff, use_equiv=use_equiv_mask)
            self.mask_focal = FocalLoss2(alpha=mask_focal_alpha, gamma=mask_focal_gamma, logits=True)

        self.eval_cfg = eval_config or {}
        self.dcc_thr = self.eval_cfg.get("dcc_thr", 4.0)
        self.decode_kw = self.eval_cfg.get("decode", {})
        self.decode_kw_mask = self.eval_cfg.get("decode_mask", {})

        self._val = self._fresh()
        self._test = {}
        self.test_results: Dict[str, SetPredResults] = {}

    @staticmethod
    def _fresh():
        return {"dets": [], "annos": [], "dcc": [], "npred": [], "ngt": []}

    # ---- one forward; returns per-node head outputs + (h, X) for the mask head ----
    def _encode(self, batch):
        h, X = self.model(batch)
        out = self.head(h, X, batch.pos)
        return out, h, X

    def _mask_loss(self, h, X, batch):
        total = h.new_zeros(())
        count = 0
        for g in range(batch.num_graphs):
            idx = (batch.batch == g).nonzero(as_tuple=True)[0]
            tgt = batch.targets[g]
            cids = torch.as_tensor(tgt["center_res_idx"], device=h.device, dtype=torch.long)
            if cids.numel() == 0:
                continue
            gt = torch.as_tensor(tgt["site_masks"], device=h.device, dtype=torch.float32)   # [K, Ng]
            ml = self.mask_head(h[idx], X[idx], batch.pos[idx], cids)                        # [K, Ng]
            dice = batched_dice_loss(ml, gt, smooth=self.mask_dice_smooth)
            bce = self.mask_focal(ml.reshape(-1), gt.reshape(-1))
            total = total + self.w_mask_dice * dice + self.w_mask_bce * bce
            count += 1
        return total / max(count, 1)

    # ---- accumulate one batch of decoded predictions vs GT ----
    def _accumulate(self, store, h, X, batch, out):
        if self.mask_mode:
            dets = decode_setpred_mask(self.mask_head, h, X, batch.pos, batch, out["q_logit"],
                                       **self.decode_kw_mask)
        else:
            dets = decode_setpred(out, batch, **self.decode_kw)
        for g, det in enumerate(dets):
            tgt = batch.targets[g]
            gt_masks = tgt["site_masks"]
            gt_cent = tgt["site_centroids"]
            store["dets"].append(det)
            store["annos"].append({"labels": np.zeros(gt_masks.shape[0], dtype=np.int64),
                                   "pocket_masks": gt_masks})
            store["npred"].append(len(det["scores"]))
            store["ngt"].append(gt_masks.shape[0])
            if len(det["centers"]) and len(gt_cent):
                d = np.linalg.norm(gt_cent[:, None, :] - det["centers"][None, :, :], axis=-1).min(axis=1)
                store["dcc"].extend(d.tolist())
            elif len(gt_cent):
                store["dcc"].extend([1e9] * len(gt_cent))

    def _reduce(self, store) -> SetPredResults:
        if not store["dets"]:
            return SetPredResults()
        ap50 = eval_map(store["dets"], store["annos"], num_classes=1, iou_thr=0.5, nproc=1)[0]
        ap30 = eval_map(store["dets"], store["annos"], num_classes=1, iou_thr=0.3, nproc=1)[0]
        dcc = np.asarray(store["dcc"], dtype=np.float64) if store["dcc"] else np.array([1e9])
        return SetPredResults(
            ap50=float(ap50), ap30=float(ap30),
            dcc_mean=float(dcc[dcc < 1e8].mean()) if (dcc < 1e8).any() else float("nan"),
            dcc_success=float((dcc < self.dcc_thr).mean()),
            num_pred=float(np.mean(store["npred"])), num_gt=float(np.mean(store["ngt"])),
        )

    # ---------------------------------------------------------------- train
    def training_step(self, batch, batch_idx):
        out, h, X = self._encode(batch)
        loss, parts = self.criterion(out, batch)
        if self.mask_mode:
            ml = self._mask_loss(h, X, batch)
            loss = loss + ml
            parts["mask"] = float(ml.detach())
        self.log("train_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        for k, v in parts.items():
            self.log(f"train_{k}", v, batch_size=batch.num_graphs)
        return loss

    # ---------------------------------------------------------------- val
    def validation_step(self, batch, batch_idx):
        out, h, X = self._encode(batch)
        loss, _ = self.criterion(out, batch)
        if self.mask_mode:
            loss = loss + self._mask_loss(h, X, batch)
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        self._accumulate(self._val, h, X, batch, out)
        return loss

    def on_validation_epoch_end(self):
        res = self._reduce(self._val)
        self.log("val_ap", res.ap50, prog_bar=True)          # monitored (AP@0.5)
        self.log("val_ap30", res.ap30, prog_bar=True)
        self.log("val_dcc_success", res.dcc_success, prog_bar=True)
        self.log("val_dcc_mean", res.dcc_mean)
        self.log("val_num_pred", res.num_pred)
        self._val = self._fresh()

    # ---------------------------------------------------------------- test
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        name = self.trainer.datamodule.get_test_loader_names()[dataloader_idx]
        if name not in self._test:
            self._test[name] = self._fresh()
        out, h, X = self._encode(batch)
        self._accumulate(self._test[name], h, X, batch, out)

    def on_test_epoch_end(self):
        for name, store in self._test.items():
            res = self._reduce(store)
            self.test_results[name] = res
            self.log(f"test_{name}_ap50", res.ap50)
            self.log(f"test_{name}_ap30", res.ap30)
            self.log(f"test_{name}_dcc_success", res.dcc_success)
            print(f"\n{'='*54}\nTEST {name}\n{'='*54}")
            print(f"  mask IoU-AP@0.5 : {res.ap50:.4f}   (UniSite ref 0.3835)")
            print(f"  mask IoU-AP@0.3 : {res.ap30:.4f}   (UniSite ref 0.5603)")
            print(f"  DCC success (<{self.dcc_thr}A): {res.dcc_success:.2%}")
            print(f"  mean DCC             : {res.dcc_mean:.3f} A")
            print(f"  avg #pred / #gt sites: {res.num_pred:.2f} / {res.num_gt:.2f}")
        self._test = {}

    def get_test_summary(self):
        return self.test_results

    # ---------------------------------------------------------------- optim
    def configure_optimizers(self):
        oc = self.hparams.optimizer_config
        opt = torch.optim.AdamW(self.parameters(), lr=oc["lr"],
                                weight_decay=oc.get("weight_decay", 1e-4))
        sc = oc.get("scheduler", {})
        sched = CosineAnnealingWarmRestarts(
            opt, T_0=sc.get("T_0", 50), T_mult=sc.get("T_mult", 1), eta_min=sc.get("eta_min", 1e-6))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
