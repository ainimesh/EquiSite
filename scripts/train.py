"""
train_setpred.py  —  end-to-end training for UniSite set-prediction (self-nominated-center)
on the GDEGAN encoder.  NEW FILE (non-destructive; does not touch train.py).

  conda run -n ainimesh python train_setpred.py --config configs/gda_hegnn_setpred.yaml
"""


import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar

from equisite.method.lightning import SetPredLightningModule
from equisite.data.dataset import create_setpred_datamodule


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config["data"].get("random_seed", 42), workers=True)

    print("--- Setting up DataModule ---")
    datamodule = create_setpred_datamodule(**config["data"])
    datamodule.setup()
    datamodule.print_dataset_summary()

    model = SetPredLightningModule(
        model_config=config["model"],
        optimizer_config=config["optimizer"],
        loss_config=config["loss"],
        head_config=config["head"],
        eval_config=config.get("eval", {}),
    )

    ckpt = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="setpred-{epoch:02d}-{val_ap:.4f}",
        monitor="val_ap", mode="max", save_top_k=3, save_last=True,
    )
    early = EarlyStopping(monitor="val_ap", mode="max",
                          patience=config["training"]["early_stopping_patience"],
                          min_delta=1e-4, verbose=True)

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto", devices=[0],
        callbacks=[ckpt, early, RichProgressBar()],
        log_every_n_steps=50,
        logger=pl.loggers.CSVLogger(save_dir=args.log_dir, name="unisite_setpred"),
        gradient_clip_val=config["training"].get("gradient_clip", 1.0),
        precision=16 if config["training"].get("mixed_precision", False) else 32,
        num_sanity_val_steps=args.sanity_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
    )

    print("--- Training ---")
    trainer.fit(model, datamodule=datamodule)
    print("--- Testing ---")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    _save_results(model, config)


def _save_results(model, config):
    res = model.get_test_summary()
    if not res:
        print("No test results.")
        return
    path = Path(config["training"].get("results_csv_path", "results/unisite_setpred.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"Dataset": k, "AP50": f"{v.ap50:.4f}", "AP30": f"{v.ap30:.4f}",
             "DCC_success": f"{v.dcc_success:.4f}", "DCC_mean": f"{v.dcc_mean:.4f}",
             "avg_pred": f"{v.num_pred:.2f}", "avg_gt": f"{v.num_gt:.2f}"} for k, v in res.items()]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print("\n" + "=" * 60 + "\nTEST RESULTS\n" + "=" * 60)
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="EquiSite/configs/gda_hegnn_setpred_maskhead.yaml")
    ap.add_argument("--checkpoint_dir", default="EquiSite/checkpoints/run")
    ap.add_argument("--log_dir", default="EquiSite/logs/")
    # dev knobs (default = full run)
    ap.add_argument("--sanity_steps", type=int, default=2)
    ap.add_argument("--limit_train_batches", type=float, default=1.0)
    ap.add_argument("--limit_val_batches", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0)
    main(ap.parse_args())
