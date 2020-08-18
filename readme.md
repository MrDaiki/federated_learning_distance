# Experiments for quality aware federated averaging

## Instalation

For running the code, you need to install python required package on python 3. 

You can setup your python virtual environment using the following command at the root of this repository:
```shell
python3 -m venv [ENV_NAME]
```
You then switch to your virtual environment using:
```shell
source [ENV_NAME]/bin/activate
```

And then install package using pip:
```shell
pip install -r requirements.txt
```

Alternatively, you can run directily install all package on your computer (which is not recommanded) by using:
```bash
pip3 install -r requirements.txt
```

## Running experiments

You can run experiments by executing  parse_compare_mmd_repartition.py, help is provided with -h option.

Warning: this experiment is very costfull in memory and cpu usage.

## Visualisation

You can run visualiez experiments by executing plot_distance_result.py, help is provided with -h option.
