

# Frozen LLMs as Map-Aware Spatio-Temporal Reasoners for Vehicle Trajectory Prediction #
![项目架构图](./fig1_resize.png)

## Installation ##


### Environment Setup ###
First, we'll create a conda environment to hold the dependencies.
```
conda env create -f env.yaml
conda activate <environment_name>
```


### Datasets ###
```
/process_data/preprocess_first_run.py 
This script preprocesses the nuScenes dataset to prepare it for our model. 
The data will be processed into .pkl and .index files, enabling lazy loading for efficient training and testing.
```

### Train ###
```
The model is trained on four 48GB GPUs, with each training session taking approximately 10 hours. We use sbatch to submit the job.
The training script can be found at: 
trajectory_prediction/train_and_test_result/trajectory_predict_train.py
```

### Test ###
```
The testing script is located at: trajectory_prediction/train_and_test_result/test_metrics.py
```


# Project Title

## Introduction
This project aims to develop a robust method for vehicle trajectory prediction using frozen LLMs (Large Language Models) integrated with map-aware spatio-temporal reasoning.

## Citations

### Papers
- **Yanjiao Liu, Jiawei Liu, Xun Gong, Zifei Nie.** "Frozen LLMs as Map-Aware Spatio-Temporal Reasoners for Vehicle Trajectory Prediction." *Proceedings of the 2026 IEEE Intelligent Vehicles Symposium (IV)*, Manuscript 216, 2026.

If you use this project or parts of it in your work, please cite the paper as follows:

```bibtex
@inproceedings{liu2026frozen,
  author    = {Yanjiao Liu and Jiawei Liu and Xun Gong and Zifei Nie},
  title     = {Frozen LLMs as Map-Aware Spatio-Temporal Reasoners for Vehicle Trajectory Prediction},
  booktitle = {Proceedings of the 2026 IEEE Intelligent Vehicles Symposium (IV)},
  year      = {2026},
  manuscript = {216}
}
