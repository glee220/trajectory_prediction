

# Frozen LLMs as Map-Aware Spatio-Tempo[Fig1.pdf](https://github.com/user-attachments/files/23538012/Fig1.pdf)
ral Reasoners for Vehicle Trajectory Prediction #

## Installation ##

### Cloning this repository ###
```
git clone --recurse-submodules <repository cloning URL>
```

### Environment Setup ###
First, we'll create a conda environment to hold the dependencies.
```
conda env create -f env.yaml
conda activate <environment_name>

```

### Data Setup ###
#### Pedestrian Datasets ####
We've already included preprocessed data splits for the ETH and UCY Pedestrian datasets in this repository, you can see them in `experiments/pedestrians/raw`. In order to process them into a data format that our model can work with, execute the follwing.
```
cd experiments/pedestrians
python process_data.py # This will take around 10-15 minutes, depending on your computer.
```

#### nuScenes Dataset ####
Download the nuScenes dataset (this requires signing up on [their website](https://www.nuscenes.org/)). Note that the full dataset is very large, so if you only wish to test out the codebase and model then you can just download the nuScenes "mini" dataset which only requires around 4 GB of space. Extract the downloaded zip file's contents and place them in the `experiments/nuScenes` directory. Then, download the map expansion pack (v1.1) and copy the contents of the extracted `maps` folder into the `experiments/nuScenes/v1.0-mini/maps` folder. Finally, process them into a data format that our model can work with.
```
cd experiments/nuScenes

# For the mini nuScenes dataset, use the following
python process_data.py --data=./v1.0-mini --version="v1.0-mini" --output_path=../processed

# For the full nuScenes dataset, use the following
python process_data.py --data=./v1.0 --version="v1.0-trainval" --output_path=../processed


## Model Training ##

### nuScenes Dataset ###
To train a model on the nuScenes dataset, you can execute one of the following commands from within the `trajectron/` directory, depending on the model version you desire.

| Model                                     | Command                                                                                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Base                                      | `python train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScenes/models/vel_ee/config.json --train_data_dict nuScenes_train_full.pkl --eval_data_dict nuScenes_val_full.pkl --offline_scene_graph yes --preprocess_workers 10 --batch_size 256 --log_dir ../experiments/nuScenes/models --train_epochs 20 --node_freq_mult_train --log_tag _vel_ee --augment`                      |
| +Dynamics Integration                     | `python train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScenes/models/int_ee/config.json --train_data_dict nuScenes_train_full.pkl --eval_data_dict nuScenes_val_full.pkl --offline_scene_graph yes --preprocess_workers 10 --batch_size 256 --log_dir ../experiments/nuScenes/models --train_epochs 20 --node_freq_mult_train --log_tag _int_ee --augment`                      |
| +Dynamics Integration, Maps               | `python train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScenes/models/int_ee_me/config.json --train_data_dict nuScenes_train_full.pkl --eval_data_dict nuScenes_val_full.pkl --offline_scene_graph yes --preprocess_workers 10 --batch_size 256 --log_dir ../experiments/nuScenes/models --train_epochs 20 --node_freq_mult_train --log_tag _int_ee_me --map_encoding --augment` |
| +Dynamics Integration, Maps, Robot Future | `python train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScenes/models/robot/config.json --train_data_dict nuScenes_train_full.pkl --eval_data_dict nuScenes_val_full.pkl --offline_scene_graph yes --preprocess_workers 10 --batch_size 256 --log_dir ../experiments/nuScenes/models --train_epochs 20 --node_freq_mult_train --log_tag _robot --incl_robot_node --map_encoding` |

In case you also want to produce the version of our model that was trained without the ego-vehicle (first row of Table 4 (b) in the paper), then run the command from the third row of the table above, but change line 132 of `train.py` to:
```
                                       return_robot=False)
```

### CPU Training ###
By default, our training script assumes access to a GPU. If you want to train on a CPU, comment out line 38 in `train.py` and add `--device cpu` to the training command.

## Model Evaluation ##
### Pedestrian Datasets ###
To evaluate a trained model, you can execute a version of the following command from within the `experiments/pedestrians` directory.
```
python evaluate.py --model <model directory> --checkpoint <epoch number> --data ../processed/<dataset>_test.pkl --output_path results --output_tag <dataset>_<vel if no integration>_12 --node_type PEDESTRIAN
```

For example, a fully-fleshed out version of this command to evaluate a model without dynamics integration for evaluation on the ETH - University scene would look like:
```
python evaluate.py --model models/eth_vel --checkpoint 100 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_12 --node_type PEDESTRIAN
```
The same for a model with dynamics integration would look like:
```
python evaluate.py --model models/eth_attention_radius_3 --checkpoint 100 --data ../processed/eth_test.pkl --output_path results --output_tag eth_12 --node_type PEDESTRIAN
```
These scripts will produce csv files in the `results` directory which can then be analyzed in the `Result Analysis.ipynb` notebook.

### nuScenes Dataset ###
If you just want to use a trained model to generate trajectories and plot them, you can do this in the `NuScenes Qualitative.ipynb` notebook.

To evaluate a trained model's performance on forecasting vehicles, you can execute a one of the following commands from within the `experiments/nuScenes` directory.

| Model                                     | Command                                                                                                                                                                                          |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Base                                      | `python evaluate.py --model models/vel_ee --checkpoint=12 --data ../processed/nuScenes_test_full.pkl --output_path results --output_tag vel_ee --node_type VEHICLE --prediction_horizon 6`       |
| +Dynamics Integration                     | `python evaluate.py --model models/int_ee --checkpoint=12 --data ../processed/nuScenes_test_full.pkl --output_path results --output_tag int_ee --node_type VEHICLE --prediction_horizon 6`       |
| +Dynamics Integration, Maps               | `python evaluate.py --model models/int_ee_me --checkpoint=12 --data ../processed/nuScenes_test_full.pkl --output_path results --output_tag int_ee_me --node_type VEHICLE --prediction_horizon 6` |
| +Dynamics Integration, Maps, Robot Future | `python evaluate.py --model models/robot --checkpoint=12 --data ../processed/nuScenes_test_full.pkl --output_path results --output_tag robot --node_type VEHICLE --prediction_horizon 6`         |

If you instead wanted to evaluate a trained model's performance on forecasting pedestrians, you can execute a one of the following.

| Model                       | Command                                                                                                                                                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Base                        | `python evaluate.py --model models/vel_ee --checkpoint=12 --data ../processed/nuScenes_test_full.pkl --output_path results --output_tag vel_ee_ped --node_type PEDESTRIAN --prediction_horizon 6`       |
| +Dynamics Integration, Maps | `python evaluate.py --model models/int_ee_me --checkpoint=12 --data ../processed/nuScenes_test_full.pkl --output_path results --output_tag int_ee_me_ped --node_type PEDESTRIAN --prediction_horizon 6` |

These scripts will produce csv files in the `results` directory which can then be analyzed in the `NuScenes Quantitative.ipynb` notebook.

## Online Execution ##
As of December 2020, this repository includes an "online" running capability. In addition to the regular batched mode for training and testing, Trajectron++ can now be executed online on streaming data!

The `trajectron/test_online.py` script shows how to use it, and can be run as follows (depending on the desired model).

| Model                                     | Command                                                                                                                                                                                             | File Changes            |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| Base                                      | python test_online.py --log_dir=../experiments/nuScenes/models --data_dir=../experiments/processed --conf=config.json --eval_data_dict=nuScenes_test_mini_full.pkl                                  | Line 110: `'vel_ee'`    |
| +Dynamics Integration                     | python test_online.py --log_dir=../experiments/nuScenes/models --data_dir=../experiments/processed --conf=config.json --eval_data_dict=nuScenes_test_mini_full.pkl                                  | Line 110: `'int_ee'`    |
| +Dynamics Integration, Maps               | python test_online.py --log_dir=../experiments/nuScenes/models --data_dir=../experiments/processed --conf=config.json --eval_data_dict=nuScenes_test_mini_full.pkl --map_encoding                   | Line 110: `'int_ee_me'` |
| +Dynamics Integration, Maps, Robot Future | python test_online.py --log_dir=../experiments/nuScenes/models --data_dir=../experiments/processed --conf=config.json --eval_data_dict=nuScenes_test_mini_full.pkl --map_encoding --incl_robot_node | Line 110: `'robot'`     |

Further, lines 145-151 can be changed to choose different scenes and starting timesteps.

During running, each prediction will be iteratively visualized and saved in a `pred_figs/` folder within the specified model folder. For example, if the script loads the `int_ee` version of Trajectron++ then generated figures will be saved to `experiments/nuScenes/models/int_ee/pred_figs/`.

## NuScenes Datasets ##

If you only want to evaluate models (e.g., produce trajectories and plot them), then the nuScenes mini dataset should be fine. If you want to train a model, then the full nuScenes dataset is required. In either case, you can find them on the [dataset website](https://www.nuscenes.org/).

