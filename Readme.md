Reduced version of 4DMOS, see [here](http://www.ipb.uni-bonn.de/pdfs/mersch2022ral.pdf)

### Pre setup
In your catkin_ws, please clone and build the following packages:
```bash
cd </path/to/catkin_ws>/src
git clone https://github.com/koide3/ndt_omp
git clone https://github.com/SMRT-AIST/fast_gicp --recursive 
git clone https://github.com/koide3/hdl_global_localization 
git clone --branch SPS https://github.com/ibrahimhroob/hdl_localization.git
```

Then build the packages:
```bash
cd </path/to/catkin_ws>
catkin build
```

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

### Dataset
Bacchus dataset:
```bash
wget https://lcas.lincoln.ac.uk/nextcloud/index.php/s/ssibg4rtrC4XFNJ/download -O Bacchus.zip && unzip Bacchus.zip && rm Bacchus.zip
```

NCLT dataset:
```bash
TODO
```

### Running
To run, export the path to the data

```bash
export DATA=path/to/dataset/sequences
```

### Training
To train a model with the parameters specified in `config/config.yaml`, run

```bash
python scripts/train.py
```

## License
This project is free software made available under the MIT License. For details see the LICENSE file.
