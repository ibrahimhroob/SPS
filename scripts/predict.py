#!/usr/bin/env python3

import torch
import click
from pytorch_lightning import Trainer

import sps.datasets.datasets as datasets
import sps.models.models as models

import torch.multiprocessing as mp

@click.command()
### Add your options here
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to checkpoint file (.ckpt) to do inference.",
    default='/sps/tb_logs/SPS_ME_Union/version_39/checkpoints/last.ckpt',
    # required=True,
)
@click.option(
    "--sequence",
    "-seq",
    type=str,
    help="Run inference on specific sequences. Otherwise, test split from config is used.",
    default= ['20220420'], #['20220420', '20220601', '20220608', '20220629', '20220714'],
    multiple=True,
)
def main(weights, sequence):
    mp.set_start_method('spawn')  # Set the start method to 'spawn' before creating any processes

    cfg = torch.load(weights)["hyper_parameters"]

    if sequence:
        cfg["DATA"]["SPLIT"]["TEST"] = list(sequence)

    cfg["TRAIN"]["BATCH_SIZE"] = 1

    # Load data and model
    cfg["DATA"]["SPLIT"]["TRAIN"] = cfg["DATA"]["SPLIT"]["TEST"]
    cfg["DATA"]["SPLIT"]["VAL"] = cfg["DATA"]["SPLIT"]["TEST"]
    data = datasets.BacchusModule(cfg)
    data.setup()

    print(len(data.test_scans))

    ckpt = torch.load(weights)
    model = models.SPSNet(cfg, len(data.test_scans))
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    model.eval()
    model.freeze()

    # Setup trainer
    trainer = Trainer(accelerator="gpu", devices=1, logger=None)

    # Infer!
    trainer.predict(model, data.val_dataloader())


if __name__ == "__main__":
    main()