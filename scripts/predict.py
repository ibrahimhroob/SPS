#!/usr/bin/env python3

import yaml
import torch
import click
from pytorch_lightning import Trainer

# import sps.datasets.datasets_nclt as datasets
import sps.datasets.blt_dataset as datasets
import sps.models.models as models

# Constants
DEFAULT_CONFIG_PATH = "./config/config.yaml"

@click.command()
### Add your options here
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to checkpoint file (.ckpt) to do inference.",
    default='/sps/best_models/420_601.ckpt',
    # required=True,
)
@click.option(
    "--sequence",
    "-seq",
    type=str,
    help="Run inference on specific sequences. Otherwise, test split from config is used.",
    default=None,
    multiple=False,
)
@click.option(
    "--config",
    "-c",
    type=str,
    help="Path to the config file (.yaml)",
    default=DEFAULT_CONFIG_PATH,
)
def main(weights, sequence, config):

    cfg = yaml.safe_load(open(config))

    if sequence:
        cfg["DATA"]["SPLIT"]["TEST"] = list(sequence)
    
    print('Test seq: ', cfg["DATA"]["SPLIT"]["TEST"])
    assert len(cfg["DATA"]["SPLIT"]["TEST"]) == 1, "Only one test SEQ is allowed at a time!"

    cfg["TRAIN"]["BATCH_SIZE"] = 1

    # Load data and model
    data = datasets.BacchusModule(cfg, test=True)
    data.setup()

    ckpt = torch.load(weights)
    model = models.SPSNet(cfg, len(data.test_scans))
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    model.eval()
    model.freeze()

    # Setup trainer
    trainer = Trainer(accelerator="gpu", devices=1, logger=None)

    # Infer!
    trainer.predict(model, data.test_dataloader())

    # Print metrics
    metrics = {
        "Loss": model.predict_loss,
        "R2": model.predict_r2,
        "dIoU": model.dIoU,
        "Precision": model.precision,
        "Recall": model.recall,
        "F1": model.F1
    }

    print('\n########## Inference Metrics ##########')
    for metric_name, metric_values in metrics.items():
        mean_value = sum(metric_values) / len(metric_values)
        space_fill = '.' * (12 - len(metric_name))  # Calculate the number of dots needed for filling
        print(f'{metric_name} {space_fill} {mean_value:.3f}')  # Fixed width for metric names
        
if __name__ == "__main__":
    main()