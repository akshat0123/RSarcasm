# Reddit Sarcasm Detection with Distilbert
This repository contains source code for detecting sarcasm in reddit comments. The dataset used in this analysis is available [here](https://www.kaggle.com/danofer/sarcasm) on kaggle.  

### Requirements
`SARC.yaml` contains a conda environment containing all required python packages needed to run the source code. `paths.json` lists all the filepaths used in running the source code. The `home` filepath must be filled in prior to running source code. The `train-balanced-sarcasm.csv` file from the [dataset](https://www.kaggle.com/danofer/sarcasm) must be placed in the `data` directory prior to running source code.  

### Source Code
- `bert.py`: Contains a wrapper class for the [huggingface](https://huggingface.co/) transformers DistilBert implementation.
- `clean.py`: Preprocessing script for the sarcasm dataset.
- `dataset.py`: Contains a `torch.utils.data.Dataset` class for the sarcasm dataset.
- `stats.py`: Script for plotting token count distribution of sarcasm dataset.
- `test.py`: Script for calculating test set accuracy.
- `train.py`: Script for fine-tuning DistilBert for sarcasm detection

### Running Source Code
Code must be run in the following order in order to produce sarcasm detection train and test results.
1. `clean.py`
2. `train.py`
3. `test.py`

Results of `train.py` and `test.py` will appear in the command prompt once the training and testing processes have completed.
