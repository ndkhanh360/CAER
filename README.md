# CAER-S
 [![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PmyTWPNCn3NNNXMIwrFh4gH0UXoPi9AJ?usp=sharing)

 This is my Pytorch implementation for the CAER-S model in the paper [Context-Aware Emotion Recognition Networks](https://caer-dataset.github.io/file/JiyoungLee_iccv2019_CAER-Net.pdf)

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
  CAER/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── configs/ - holds configuration for training and testing
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
Download [CAER-S dataset](https://caer-dataset.github.io/download.html) and the results of `dlib's cnn detector` for the [training](https://drive.google.com/file/d/1Em4LUdJ6VS8sOo_XdBi96xu8nwoo_Y4g/view?usp=sharing), [validation](https://drive.google.com/file/d/1thrfz6IdPIXSQRZ6LW-5tGVabbfHUoeA/view?usp=sharing) and [test set](https://drive.google.com/file/d/1eEpjAmLrz9f9_SqZh5qCPsScHikg91V4/view?usp=sharing), then put them into folder `data`.

## Usage 
Move into `configs` directory and create configuration file for training and/or testing:
```
{
  // configuration for both training and testing
  "name": "CAERS_Session",             // session name
  "n_gpu": 1,                          // number of GPUs to use for training.
  "arch": {                            // architecture
      "type": "CAERSNet",
      "args": {}
  },
  "loss": "cross_entropy",             // loss
  "metrics": [
    "accuracy"                         // list of metrics to evaluate
  ],   

  // configuration for testing only 
  "test_loader": {
      "type": "CAERSDataLoader",
      "args":{
          "root": "data/CAER-S/test",
          "detect_file": "data/test.txt",
          "train": false,
          "batch_size": 128,
          "shuffle": false,
          "num_workers": 16
      }
  },

  // configuration for training only
  "train_loader": {                    // loader for training 
    "type": "CAERSDataLoader",         // selecting data loader
    "args":{
      "root": "data/CAER-S/train",     // train dataset directory
      "detect_file": "data/train.txt", // face detector results
      "batch_size": 128,               // batch size
      "shuffle": true,                 // shuffle training data 
      "num_workers": 16,               // number of cpu processes to be used for data loading
    }
  },

  "val_loader": {
    "type": "CAERSDataLoader",
    "args":{
        "root": "data/CAER-S/test",
        "detect_file": "data/val.txt",
        "train": false,
        "batch_size": 128,
        "shuffle": false,
        "num_workers": 16
    }
  },

  "optimizer": {
    "type": "SGD",
    "args":{
      "lr": 1e-2,                      // learning rate
      "momentum": 0.9,                 // (optional) weight decay
      "nesterov": true
    }
  },                        
  "trainer": {
    "epochs": 50,                      // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 10,                   // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "max val_accuracy"      // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 20	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```

Once you've finished configuration, enter this snippet to train the model
```
python train.py --config [train config path]
```
To resume training from earlier checkpoint, add `--resume` flag
```
python train.py --resume [your checkpoint path]
```
To evaluate the model on test data, simply enter
```
python test.py --config [test config path] --resume [your checkpoint path]
```

You can download the pretrained CAER-S model which achieves a test accuracy of **76.81%** at [this link](https://drive.google.com/file/d/1HxHZQmWnXbhYV0_HU2q-2fcC3twtGJZp/view?usp=sharing).

## Acknowledgements
Many thanks to Victor Huang for an amazing [Pytorch-template](https://github.com/victoresque/pytorch-template).