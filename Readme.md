<div align="center">
  <h1>Generalizable Stable Points Segmentation for 3D LiDAR Scan-to-Map Long-Term Localization</h1>

<p>
  <img src="https://github.com/ibrahimhroob/SPS/assets/47870260/9ea7cd26-9db5-4ba2-bca2-033df59e7b26" width="600"/>
</p>

<p>
  <i>Our method segments stable and unstable points in 3D LiDAR scans exploiting the discrepancy of scan voxels and overlapping map voxels (highlighted as submap voxels). We showcase two LiDAR scans captured during separate localization sessions within an outdoor vineyard. The scan on the left depicts the vineyard state in April, while the scan on the right reveals environmental changes in plant growth in June</i>
</p>

<details>
<summary><b>Click here for qualitative results!</b></summary>
  
[![ScanSPS](https://github.com/ibrahimhroob/SPS/assets/47870260/0f93743f-170c-4ca3-be15-77623f45720c)](https://github.com/ibrahimhroob/SPS/assets/47870260/0f93743f-170c-4ca3-be15-77623f45720c)


 <i>Our stable points segmentation prediction for three datasets. The stable points are depicted in black, while the unstable points are represented in red.</i>

</details>

</div>


### Building the Docker image
We provide a ```Dockerfile``` and a ```docker-compose.yaml``` to run all docker commands. 

**IMPORTANT** To have GPU access during the build stage, make ```nvidia``` the default runtime in ```/etc/docker/daemon.json```:

    ```yaml
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            } 
        },
        "default-runtime": "nvidia" 
    }
    ```
    Save the file and run ```sudo systemctl restart docker``` to restart docker.


To build the image, simply type the following in the terminal:
```bash
bash build_docker.sh
```

Once the build process finishes, initiate the Docker container in detached mode using Docker Compose from the project directory:
```bash
docker-compose up -d # or [docker compose up -d] for older versions
```

## Usage Instructions

### Training

To train the model with the parameters specified in `config/config.yaml`, follow these steps:

1. Export the path to the dataset (This step may be necessary before initiating the container):
    ```bash
    export DATA=path/to/dataset
    ```

2. Initiate training by executing the following command from within the container:
    ```bash
    python scripts/train.py
    ```

### Segmentation Metrics

To evaluate the segmentation metrics for a specific sequence:

```bash
python scripts/predict.py -seq <SEQ ID>
```

This command will generate reports for the following metrics: 
- uIoU (unstable points IoU)
- Precision
- Recall
- F1 score

### Data
You can download the post-processed and labeled [BLT dataset](https://drive.google.com/file/d/1beRMNbg2sRSOzMpRuI8Eh409girAdlck/view?usp=drive_link) and the parking lot of [NCLT dataset](https://drive.google.com/file/d/16T-EkoZnDHH4xIIj7PKuNXJ_LMl9x3Pm/view?usp=drive_link) from the proveded links.

The [weights](https://drive.google.com/file/d/1Ic80AvYh9Jf77cBMp2SXy8y7y-YExprC/view?usp=drive_link) of our pre-trained model can be downloaded as well.

Here the general structure of the dataset: 
```
DATASET/
├── maps
│   ├── base_map.asc
│   ├── base_map.asc.npy
│   └── base_map.pcd
└── sequence
    ├── SEQ
    │   ├── map_transform
    │   ├── poses
    |   |   ├── 0.txt
    |   |   └── ...
    │   └── scans
    |       ├── 0.npy
    |       └── ...
    |
    └── ...
```

## Publication
If you use our code in your academic work, please cite the corresponding [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/hroob2024ral.pdf):

```bibtex
@article{hroob2024ral,
  author = {I. Hroob* and B. Mersch* and C. Stachniss and M. Hanheide},
  title = {{Generalizable Stable Points Segmentation for 3D LiDAR Scan-to-Map Long-Term Localization}},
  journal = {IEEE Robotics and Automation Letters (RA-L)},
  volume = {9},
  number = {4},
  pages = {3546-3553},
  year = {2024},
  doi = {10.1109/LRA.2024.3368236},
}
```

## Acknowledgments
This implementation is inspired by [4DMOS](https://github.com/PRBonn/4DMOS).


## License
This project is free software made available under the MIT License. For details see the LICENSE file.
