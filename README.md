

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


