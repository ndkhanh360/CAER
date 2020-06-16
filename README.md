<<<<<<< HEAD
# CAER-S
 [![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
 
 This is our Pytorch implementation for the baselines in the paper [Context-Aware Emotion Recognition Networks](https://caer-dataset.github.io/file/JiyoungLee_iccv2019_CAER-Net.pdf)

## Installation
First, clone this repository
```bash
git clone https://github.com/ndkhanh360/CAER.git
```
Then, install the dependencies
```bash
pip install -r requirements.txt
```
## Folder Structure
This project was created with [Pytorch-template](https://github.com/victoresque/pytorch-template) by Victor Huang. It has the following structure
  ```
  Baseline/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```
Download [CAER-S dataset](https://caer-dataset.github.io/download.html) and the results of `dlib's cnn detector` for the [training set](https://drive.google.com/file/d/1CKRAnXeGtqoGm77jjJdFhTHUuH12mo3s/view?usp=sharing) and [test set](https://drive.google.com/file/d/16HGNxB_rsOqp1dNeYhpQ3SY8oOURpszC/view?usp=sharing), then put them into folder `data`.

## Usage 
Move into `Baseline` directory and modify `config.json` file
```
{
  "name": "ResNet_Session",            // training session name
  "n_gpu": 1,                          // number of GPUs to use for training.
  
  "arch": {
    "type": "ResNet",                  // name of model architecture ("ResNet", "VGGNet", "AlexNet")
    "args": {
      "fine_tune": false,              // whether or not you want to fine tune the whole model
      "drop_out": false                // whether or not you want to add a dropout layer before the final FC
    }                
  },
  "train_loader": {                    // loader for training 
    "type": "CAERSDataLoader",         // selecting data loader
    "args":{
      "root": "data/CAER-S/train",     // train dataset directory
      "detect_file": "data/train.txt", // face detector results (delete to input the whole image instead)
      "batch_size": 32,                // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "validation_split": 0.125        // size of validation dataset. float(portion) or int(number of samples)
      "num_workers": 2,                // number of cpu processes to be used for data loading
    }
  },
  "test_loader": {                     // loader for testing 
    "type": "CAERSDataLoader",         // selecting data loader
    "args":{
      "root": "data/CAER-S/test",      // train dataset directory
      "detect_file": "data/test.txt",  // face detector results (delete to input the whole image instead)
      "batch_size": 32,                // batch size
      "shuffle": false,                // do not shuffle test data
      "num_workers": 2,                // number of cpu processes to be used for data loading
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 1e-4,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "cross_entropy",             // loss
  "metrics": [
    "accuracy"                         // list of metrics to evaluate
  ],                         
  "trainer": {
    "epochs": 50,                      // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 2,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 5	                   // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```
**Note**: the following options are the most important ones that need to be modified
* `arch` -> `type`, `fine_tune`
* `train_loader` -> `args` -> `detect_file`
* `test_loader` -> `args` -> `detect_file`

Once you've finished configuration, enter this snippet to train the model
```
python train.py --config config.json
```
To resume training from earlier checkpoint, add `--resume` flag
```
python train.py --config config.json --resume [your checkpoint path]
```
To evaluate the model on test data, simply enter
```
python test.py --config config.json --resume [your checkpoint path]
```
The checkpoint file corresponding to the lowest validation loss is located inside `saved/models/<checkpoint dir>` directory with the name `model_best.pth`. 

## Acknowledgements
Many thanks to Victor Huang for an amazing [Pytorch-template](https://github.com/victoresque/pytorch-template).