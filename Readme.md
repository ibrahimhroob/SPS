Reduced version of 4DMOS, see [here](http://www.ipb.uni-bonn.de/pdfs/mersch2022ral.pdf)

### Pre setup
Clone the following packages into `c_ws/src`
```bash
git clone https://github.com/koide3/ndt_omp
git clone https://github.com/SMRT-AIST/fast_gicp --recursive 
git clone https://github.com/koide3/hdl_global_localization 
git clone --branch SPS https://github.com/ibrahimhroob/hdl_localization.git
git clone https://github.com/ibrahimhroob/sps_filter.git
```

### Without Docker
Without Docker, you need to install the dependencies specified in the `setup.py`.This can be done in editable mode by running

```bash
python3 -m pip install --editable .
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
