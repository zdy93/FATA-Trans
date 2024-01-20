# FATA-Trans
This repository is the official implementation of the **CIKM2023** paper ["FATA-Trans: Field And Time-Aware Transformer for Sequential Tabular Data"](https://doi.org/10.1145/3583780.3614879). Some code scripts are adapted from [Tabular Transformers for Modeling Multivariate Time Series](https://github.com/IBM/TabFormer#tabular-transformers-for-modeling-multivariate-time-series).

## Requirement
### Language
* Python3 >= 3.9
### Modules
* torch==1.13.0
* transformers==4.26.1
* tqdm==4.64.1
* scikit-learn==1.2.0
* matplotlib==3.6.2
* numpy==1.24.2
* pandas==1.1.5
These packages can be installed directly by running the following command:
```
pip install -r requirements.txt
```
## Dataset
### Synthetic Transaction
Synthetic transaction dataset is provided in the [TabFormer github repoistory](https://github.com/IBM/TabFormer). 
### Amamzon Product Reviews
Amazon product reviews datasets are available at [here](https://nijianmo.github.io/amazon/index.html). In our paper, we used the "5-core" subsets.

## Running
1. run [preprocess_IBM_v2.ipynb](./preprocess_IBM_v2.ipynb) or [preprocess_amazon_liang.ipynb](./preprocess_amazon_liang.ipynb) to split the dataset raw files into train/val/test csv files.
2. run [preload_dataset.ipynb](./preload_dataset.ipynb) to excute the first stage processing.
3. run either [process_IBM_dataset.ipynb](./process_IBM_dataset.ipynb) or [process_amazon_dataset.ipynb](./process_amazon_dataset.ipynb) to get the model-specific dataset.
4. run files named as "run_main_....ipynb" to pretrain, finetune, train from scratch, or expert embeddings from a model. (You can also directly run with [main_ibm.py](./main_ibm.py) or [main_amazon.py](./main_amazon.py)).

Linux bash scripts under the directory [sh_commands](./sh_commands) can be used to run these jupyter notebooks mentioned above with the Python module papermill (we used the version 2.4.0). For model or dataset specfic settings, you are reffered to these bash scripts.
