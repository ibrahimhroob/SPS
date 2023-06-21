Reduced version of 4DMOS, see [here](http://www.ipb.uni-bonn.de/pdfs/mersch2022ral.pdf):

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
