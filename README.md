# YOLO training using Pseudo labeled data

The repository contains code for training the YOLO model on various markup (original data, pseudo labelled and variations).


## Datasets

I use PASCAL VOC 2007 Dataset. The PASCAL VOC (Visual Object Classes) dataset is a well-known object detection, segmentation, and classification dataset. It is designed to encourage research on a wide variety of object categories and is commonly used for benchmarking computer vision models. It is an essential dataset for researchers and developers working on object detection, segmentation, and classification tasks.

## Instructions

**Installation:**

1.Clone the repository from GitHub.

```bash
git clone git@github.com:NikolayKn/Pseudo_label_YOLO.git
```

2. Install all dependencies using pip.

```bash
pip install -r requirements.txt
```

**Downloading VOC2007 dataset:**

1.Activate the environment.

```bash
source .venv/bin/activate
```

2. Run the ```download_VOC.py``` script to download the VOC2007 Dataset.

```bash
python download_VOC.py
```

**Using Notebooks:**

1. In the `notebooks` directory, you will find various Jupyter notebooks for different purposes:
   - **Visualization of Results:** Notebooks with `visualizations` in their titles are used for visualizing the results of various experiments.
   - **Configuration Files Creation:** These notebooks help in creating configuration `.yaml` files for setting up and organizing different experiment runs.
   - **Playground:** A notebook named `playground` is available for testing various approaches and constructs.

**Configurations:**

1. The `configs` directory contains various configuration files for different experiments.

**Source Directory:**

1. The `source` directory contains the following scripts:
   - **run_experiment.py:** Script for running experiments. Example usage:
   
     ```bash
     python run_experiments.py configs/cfg.yaml
     ```
   
   - **run_experiment_only_train.py:** Similar to `run_experiment.py`, but only performs training without running tests on the test dataset.
   
   - **run_experiment_test_all_weights.py:** Adds the ability to train on a partial dataset and test all saved checkpoints during training.



## Credits

- [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/)
- [ mAP realization by Ultralitics](https://github.com/ultralytics/yolov5/blob/master/val.py)  
