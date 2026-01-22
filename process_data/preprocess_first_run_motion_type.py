import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from torch.utils.data import Dataset
from accelerate import Accelerator
from nuscenes.map_expansion.map_api import NuScenesMap
from environment.map import GeometricMap
from tqdm import tqdm
import dill
from nuscenes.nuscenes import NuScenes
from precess_data.motion_type_classifier import classify_motion_type

class Dataset_Nuscene(Dataset):
    def __init__(self, 
                 flag='train', 
                 load_mode = 'social', 
                 observe_len=4, pred_len=12, time_step=0.5,
                #  data_path="../dataFactory/nuScenes",
                 data_path="/home/liuyanjiao/trajectory_predict/dataFactory/nuScenes",
                 allow_incomplete_traces=False, 
                 allow_invisible_traces=False,
                 allow_pedestrian_traces=False, 
                 nbr_dis_threshold = 50,
                 accelerator = Accelerator()):
        """
        :param data_path: 数据集存放的位置
        :param flag: "train", "val" or "test"
        :param load_mode, "debug" or "social" or "map"
        :param observe_len: 观测frame的个数
        :param pred_len: 预测frame的个数
        :param time_step: 相临frame之间的时间间隔
        :param allow_incomplete_traces: 是否允许segment（观测+预测frame）中出现不完整的轨迹
        :param allow_pedestrian_traces: 数据集是否涵盖行人的轨迹
        :param nbr_dis_threshold: 定义agent相连时的最大距离
        """

        self.data_path = data_path 
        self.accelerator = accelerator
        self.observe_len = observe_len
        self.pred_len = pred_len

        # init
        assert flag in [ 'test', 'val','train']
        self.flag = flag

        assert load_mode in ['debug', 'social', 'map']
        self.load_mode = load_mode

        self.nbr_dis_threshold = nbr_dis_threshold

        self.time_step = time_step
        self.allow_incomplete_traces = allow_incomplete_traces
        self.allow_pedestrian_traces = allow_pedestrian_traces
        self.allow_invisible_traces = allow_invisible_traces
        self.feature_dimension = 5

        # 获取scene-map之间的对应关系
        self.map_name_path = os.path.join(self.data_path, "map_name.txt")
        # 'scene_map' is a dict, specify the maps from scene to map name
        self.scene_map = self.get_scene_map(self.map_name_path)
        ## 加个地图层
        self.nuscene_layers_map = self.get_nuscene_layers_map(self.data_path)
        # Default orientation where North is up
        self.patch_angle = 0  
        self.homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]]) 
        self.layer_names = ['lane', 'road_segment', 'drivable_area', 'road_divider', 'lane_divider', 'stop_line',
                   'ped_crossing', 'stop_line', 'ped_crossing', 'walkway']
        # self.nusc = NuScenes(version='v1.0-trainval', verbose=False, dataroot="../dataFactory/v1.0-trainval_meta")
        self.nusc = NuScenes(version='v1.0-trainval', verbose=False, dataroot="/home/liuyanjiao/trajectory_predict/dataFactory/v1.0-trainval_meta")

        self.id2class = {1: "small vehicle", 2: "big vehicle", 3: "pedestrian",
                         4: "motorcyclist and bicyclist", 5: "others"}

        # 读取数据
        print('===============Loading and parsing data...===============')
        self.__read_data__()
        self.__social_load__()
        print('=========================Done!===========================')

    def __read_data__(self):  

        # return_file_path = f'./precess_data/return_data_list_{self.flag}.pkl'
        return_file_path = f'./type_{self.flag}.pkl'

        if not os.path.exists(return_file_path):   
                
            # 确定数据集文件夹路径
            self.data_path = os.path.join(self.data_path, f"prediction_{self.flag}")    
            # 建立list，缓存所有数据
            self.scene_segment_datalist = []
            # 获取相应 (train, or val, or test)数据集文件下的目录
            files = os.listdir(self.data_path)

            # 第一层遍历：场景
            for filename in tqdm(files):
                calc_delta_x_y = 0
                # 偏移量
                delta_x_y = (0,0)            

                if filename.split('.')[-1] != "txt":
                    continue
                file_path = os.path.join(self.data_path, filename)
                scene_name = filename.split('.')[0] #'scene-00xx'

                # np.genfromtxt可把txt内容直接生成矩阵，delimiter定义了矩阵元素的 分割符号
                # data大小为(一个场景内样本点个数，10)， 分别代表frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading
                # 关于object type-{1:small vehicle, 2:big vehicle, 3:pedestrian,4:motorcyclist and bicyclist, 5:others}
                data = np.genfromtxt(fname=file_path, delimiter=" ")

                # 滤除数据集中object_type = 5的类（others 类）
                data = data[~(data[:, 2] == 5)]

                # 将position_x, position_y分别堆叠起来，得到一个(n,2)的矩阵得到一个scene的最值
                find_min_max = data[~(data[:, 2] == 4)]
                position_stack = np.stack([find_min_max[:, 3], find_min_max[:, 4]], axis=1)
                min_x,min_y = np.min(position_stack, axis=0)
                max_x,max_y = np.max(position_stack, axis=0)

                # 确定场景初始时刻的id
                start_frame_id = int(np.min(data[:, 0]))
                # 场景内Frames个数
                numFrames = len(np.unique(data[:, 0]))

                # numSlices代表从当前scene文件中可以获得的有效数据片段个数
                # 计算方法为 总frame数 - 任务覆盖的frame数 + 1
                numSlices = numFrames - (self.observe_len + self.pred_len) + 1

                # 记录序列id
                current_frame = -1

                # 第二层遍历：Segment，Segment覆盖的frame长度是 seq_len + pred_len
                for slice_id in range(numSlices):                              
                    current_frame += 1
                    scene_segment_data = {
                        "init_frame_id": current_frame,
                        "observe_length": self.observe_len,
                        "predict_length": self.pred_len,
                        "time_step": self.time_step, 
                        "feature_dimension": self.feature_dimension,
                        "objects": {},
                        "map_name": self.scene_map[scene_name],
                        "scene_name": scene_name,   
                        "delta_x_y":delta_x_y,
                        "scene_point_min_max":[min_x,min_y,max_x,max_y],               
                    }

                    # 第三层遍历：frame, 对scene_segment_data 进行填充
                    for local_frame_id in range((self.observe_len + self.pred_len)):

                        # frame_id是local_frame_id在scene中的绝对id
                        frame_id = start_frame_id + slice_id + local_frame_id
                        # frame_data是在frame_id下存在的全部样本点，数量于当前时刻下场景中的运动物体数量一致
                        frame_data = data[data[:, 0] == frame_id, :]

                        # 第四层遍历：object
                        for obj_index in range(frame_data.shape[0]):

                            # 获取当前frame下，某个object的state
                            obj_data = frame_data[obj_index, :]

                            # 该object的id
                            obj_id = str(int(obj_data[1]))
                            # 该obj的种类
                            obj_type = int(obj_data[2])

                            # 忽略object_type = 4 (自行车，摩托车) 和 object_type = 5（其他类）的类别
                            if self.allow_pedestrian_traces:
                                type_threshold = 3
                            else:
                                type_threshold = 2
                            if obj_type > type_threshold:
                                continue

                            ###################################################################################
                            # 当object从未被记录过时，将其添加到scene_segment_data的"object"中
                            if obj_id not in scene_segment_data["objects"]:
                                # 以下的if判断语句说明只有出现在观测历史中的object才会被记录，出现在当前时刻或者未来时刻的object是不会被记录的
                                if local_frame_id < self.observe_len:
                                    scene_segment_data["objects"][obj_id] = {
                                        "type": obj_type,
                                        "complete": True,  # 这个标志位指示某些轨迹是否是完整的，某些frame是否存在缺失
                                        "visible": True,  # 这个标志观测轨迹的最后一个frame是否可见
                                        "observe_trace": np.zeros((self.observe_len, 2)),
                                        "observe_feature": np.zeros((self.observe_len, self.feature_dimension)),
                                        "observe_mask": np.zeros(self.observe_len),
                                        "future_trace": np.zeros((self.pred_len, 2)),
                                        "future_feature": np.zeros((self.pred_len, self.feature_dimension)),
                                        "predict_trace": np.zeros((self.pred_len, 2)),
                                        "future_mask": np.zeros(self.pred_len),
                                        "nbr_agent": [],
                                    }
                                else:
                                    continue
                            ###################################################################################

                            # 记录位置
                            obj = scene_segment_data["objects"][obj_id]
                            if local_frame_id < self.observe_len:
                                obj["observe_trace"][local_frame_id, :] = obj_data[3:5]  # x,y数据
                                obj["observe_feature"][local_frame_id, :] = obj_data[
                                                                            5:]  # position_z, object_length, object_width, object_height, heading
                                obj["observe_mask"][local_frame_id] = 1  # 有观测值为1，无观测值为0
                            else:
                                obj["future_trace"][local_frame_id - self.observe_len, :] = obj_data[3:5]
                                obj["future_feature"][local_frame_id - self.observe_len, :] = obj_data[5:]
                                obj["future_mask"][local_frame_id - self.observe_len] = 1


                    ########## 获取每个agent周围的T时刻周围的agent
                    objects = scene_segment_data['objects']
                    for _key, _value in objects.items():
                        if _value['observe_mask'][-1] != 1.:
                            continue
                        av_origin = _value['observe_trace'][-1]
                        for key_, value_ in objects.items():
                            if (value_['observe_mask'][-1] != 1.) or (_key == key_):
                                continue
                            dis = value_['observe_trace'][-1] - av_origin
                            dis = np.sqrt(np.sum(dis ** 2))
                            if dis <= self.nbr_dis_threshold:
                                _value['nbr_agent'].append(key_)

                    # 对scene_segment_data进行筛选，避免无效信息的产生
                    scene_segment_data = self.scene_segment_filter(scene_segment_data)

                    if scene_segment_data == None:
                        continue

                    # 计算偏移量（x,y） 每个scene 计算一次
                    if calc_delta_x_y == 0:
                        ego_traj = scene_segment_data["objects"]['0']["observe_trace"][0:4]
                        delta_x_y = self.delta_recog(np.array(ego_traj), scene_name, self.nusc)                  
                        min_x += delta_x_y[0]
                        min_y += delta_x_y[1]
                        max_x += delta_x_y[0]
                        max_y += delta_x_y[1]
                        calc_delta_x_y = 1
                    scene_segment_data['delta_x_y'] = delta_x_y
                    scene_segment_data['scene_point_min_max'] = [min_x,min_y,max_x,max_y]   
                    self.scene_segment_datalist.append(scene_segment_data)
            print("scene_segment_datalist processing is done!")

    def __social_load__(self):        
        self.with_map_path =  f'./type_{self.flag}.pkl'   
        self.index_path = self.with_map_path.replace('.pkl', '.index') 
        
        if not os.path.exists(self.with_map_path):
                  
            agent_data_list = []
            try:
                for scene_segment in tqdm(self.scene_segment_datalist):            
                    map_name = scene_segment['map_name']
                    scene_name = scene_segment['scene_name']
                    obj_id_list = scene_segment['objects'].keys()
                    scene_segment_data = scene_segment['objects']
                    delta_x_y = scene_segment['delta_x_y']
                    x_min,y_min,x_max,y_max = scene_segment['scene_point_min_max']
                                    
                    # a = time.time()
                    #------------ 添加小地图 start----------------#
                    # 中心点 x y height width
                    x_min_map = x_min - 50
                    y_min_map = y_min - 50
                    x_max_map = x_max + 50
                    y_max_map = y_max + 50
                    x_size = x_max_map - x_min_map
                    y_size = y_max_map - y_min_map
                    patch_box = (x_min_map + 0.5 * (x_max_map - x_min_map), y_min_map + 0.5 * (y_max_map - y_min_map), y_size, x_size)
                    patch_angle = 0  # Default orientation where North is up
                    canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
                    map_api = self.nuscene_layers_map[self.scene_map[scene_name]]
                    map_mask = (map_api.get_map_mask(patch_box, patch_angle, self.layer_names, canvas_size) * 255.0).astype(np.uint8)
                    map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
                    # VEHICLES
                    map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)
                    vehicle_map = GeometricMap(data=map_mask_vehicle, homography=self.homography, description=', '.join(self.layer_names))

                    map_mask_plot = np.stack(((np.max(map_mask[:3], axis=0) - (map_mask[3] + 0.5 * map_mask[4]).clip(
                        max=255)).clip(min=0).astype(np.uint8), map_mask[8], map_mask[9]), axis=0)
                    visualization_map = GeometricMap(data=map_mask_plot, homography=self.homography, description=', '.join(self.layer_names))
                    #------------ 添加segment小地图 end  ----------------#  
                    # b = time.time()
                    # print("one segment map cost time: ", b-a, "s")

                    ####### obj坐标移动到与地图坐标对应的点上
                    # 相对偏移量
                    # (x_map,y_map) = (x,y) + (delta_x_y) - (x_min,y_min) + offset
                    offset_x_y = np.array([delta_x_y[0], delta_x_y[1]]) - np.array([x_min, y_min]) + np.array([50, 50])
                    for obj_id in obj_id_list:
                        obj_segment_data = scene_segment_data[obj_id]
                        ######### 在进行训练的过程中不考虑不完整的agent，但nbr_agent可以考虑incomplete但visible的agent
                        if obj_segment_data['complete'] == False:
                            continue
                        ######### 在进行训练的过程中不考虑行人轨迹的预测
                        if obj_segment_data['type'] >= 3:
                            continue
                        
                        # ####### 计算旋转角
                        # heading_vector = obj_segment_data['observe_trace'][-1] - obj_segment_data['observe_trace'][-2]
                        # # 旋转角 北是0 统一朝x正方向转
                        # rotate_angle = -np.arctan2(heading_vector[1], heading_vector[0])* 180 / np.pi;
                        ####### 计算旋转角
                        heading_vector = obj_segment_data['observe_trace'][-1] - obj_segment_data['observe_trace'][-2]
                        # 旋转角 北是0 统一朝x正方向转
                        rotate_angle = np.arctan2(heading_vector[0], heading_vector[1])* 180 / np.pi;

                        ####### 计算旋转矩阵 标准型 
                        rotate_mat = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)],
                                            [np.sin(rotate_angle), np.cos(rotate_angle)]])
                        # # 将90-angle带入得到这个旋转矩阵
                        # rotate_mat = np.array([[np.sin(rotate_angle), -np.cos(rotate_angle)],
                        #                        [np.cos(rotate_angle),  np.sin(rotate_angle)]])

                        ####### nbr坐标平移
                        nbr_agent_observe_trace = [
                            scene_segment_data[agent_id]['observe_trace'] + offset_x_y 
                            for agent_id in obj_segment_data['nbr_agent']
                        ]
                        nbr_agent_observe_trace = np.stack(nbr_agent_observe_trace, axis=0) if nbr_agent_observe_trace else np.empty((0,))
                        
                        nbr_agent_position = [
                            np.concatenate([scene_segment_data[agent_id]['observe_trace'] + offset_x_y,
                                            scene_segment_data[agent_id]['future_trace'] + offset_x_y], axis=0)
                            for agent_id in obj_segment_data['nbr_agent']
                        ]
                        nbr_agent_position = np.stack(nbr_agent_position, axis=0) if nbr_agent_position else np.empty((0,))

                        nbr_agent_mask = [
                            np.concatenate([scene_segment_data[agent_id]['observe_mask'],
                                            scene_segment_data[agent_id]['future_mask']], axis=0)
                            for agent_id in obj_segment_data['nbr_agent']
                        ]
                        # 类型
                        # 传入全局坐标系坐标
                        motion_type = classify_motion_type(obj_segment_data['observe_trace'] + delta_x_y,
                                                            obj_segment_data['future_trace'] + delta_x_y, 
                                                            map_api, 
                                                            vehicle_map)
                        # print("motion_type :",motion_type)
                        obj_centric_dict = {
                            'id': str(obj_id),
                            'type': obj_segment_data['type'],
                            'observe_trace': obj_segment_data['observe_trace'] + offset_x_y,
                            'future_trace': obj_segment_data['future_trace'] + offset_x_y,
                            'position': np.concatenate([obj_segment_data['observe_trace'] + offset_x_y, obj_segment_data['future_trace'] + offset_x_y], axis=0),#position @ rotate_mat.T 
                            'agent_origin': obj_segment_data['observe_trace'][-1] + offset_x_y,
                            'heading_vector': heading_vector,
                            'rotate_angle': rotate_angle,
                            'rotate_mat': rotate_mat.T,
                            'nbr_agent': obj_segment_data['nbr_agent'],
                            'nbr_agent_type': [scene_segment_data[i]['type'] for i in obj_segment_data['nbr_agent']],
                            'nbr_agent_observe_trace': nbr_agent_observe_trace,
                            'nbr_agent_position': nbr_agent_position,
                            'nbr_agent_mask': nbr_agent_mask,
                            'map_name':map_name,
                            'scene_name':scene_name,
                            'vehicle_map': vehicle_map,
                            'visualization_map': visualization_map,
                            'scene_point_min_max' :scene_segment['scene_point_min_max'],
                            'delta_x_y': delta_x_y,
                            'offset_x_y': offset_x_y,
                            'motion_type' : motion_type,                           

                        }
                        agent_data_list.append(obj_centric_dict)        
            except Exception as e:
                print(f"An error occurred: {e}")
                
            finally:

                if len(agent_data_list) > 0:
                    try:
                        # 置空偏移表
                        self.return_data_offsets = []

                        # 边 dump 边记录当前位置
                        with open(self.with_map_path, 'wb') as pf:
                            for item in agent_data_list:
                                self.return_data_offsets.append(pf.tell())
                                dill.dump(item, pf, protocol=dill.HIGHEST_PROTOCOL)

                        # 写出 .index（每行一个 offset）
                        with open(self.index_path, 'w') as f_idx:
                            f_idx.write('\n'.join(map(str, self.return_data_offsets)))

                        print("Data successfully saved.")
                    except Exception as save_error:
                        print(f"Failed to save data: {save_error}")
        else:
            print(f"已经生成了{self.flag}.pkl文件")

    def scene_segment_filter(self, scene_segment_data):

        # 完成了对scene_segment_data的填充，现在对其进行filter
        # invalid_obj_ids:在观测时域完全未出现的agent， 
        # complete_obj_ids：观测时域与预测时域的所有frame中都有数据，
        # visible_obj_ids:在当前时刻可以观察到的agent
        invalid_obj_ids = []
        complete_obj_ids = []
        visible_obj_ids = []

        for obj_id, obj in scene_segment_data["objects"].items():

            # 记录在历史观测过程中完全未被观测到的objetc_id
            if np.sum(obj["observe_mask"]) <= 0:
                invalid_obj_ids.append(obj_id)

            # 如果allow_incomplete_traces=False，代表历史观测+未来预测中，一帧的数据都不可以缺失
            if np.min(np.concatenate((obj["observe_mask"], obj["future_mask"]), axis=0)) <= 0:
                if not self.allow_incomplete_traces:
                    invalid_obj_ids.append(obj_id)
                else:
                    obj["complete"] = False
            elif obj_id not in invalid_obj_ids:
                complete_obj_ids.append(obj_id)

            if np.min(obj["observe_mask"][-1]) <= 0:
                if not self.allow_invisible_traces:
                    invalid_obj_ids.append(obj_id)
                else:
                    obj["visible"] = False
            elif obj_id not in invalid_obj_ids:
                visible_obj_ids.append(obj_id)

        for invalid_obj_id in invalid_obj_ids:
            if invalid_obj_id in scene_segment_data["objects"].keys():
                del scene_segment_data["objects"][invalid_obj_id]
        # may create empty data, especially after invalid data removal
        if len(scene_segment_data["objects"]) == 0:
            return None
        else:
            return scene_segment_data
    def get_scene_map(self, map_name_path):
        '''
        :param map_name_path: str, the path of 'AdvTra/dataset/nuScenes/map_name.txt'
        :return: dict, {'scene-0010':'singapore-onenorth', 'scene-0164':'boston-seaport', ....}
        '''
        base_dir = os.path.dirname(os.path.abspath(__file__))  
        map_path = os.path.normpath(os.path.join(base_dir, '..', 'dataset', 'nuScenes', 'map_name.txt'))
        scene_map = {}
        with open(map_name_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                tokens = line[:-1].split(' ')
                scene_name, map_name = tokens[1], tokens[2]  # i.e., scene_name = 'scene-0010'; map_name='singapore-onenorth'
                scene_map[scene_name] = map_name
        return scene_map
    def get_nuscene_layers_map(self, map_name_path):
        # 大地图 singapore-onenorth boston-seaport singapore-queenstown singapore-hollandvillage
        map_dict = {}
        for i in set(self.scene_map.values()):
            map_dict[i] = NuScenesMap(dataroot=map_name_path, map_name=i)
        return map_dict
    def delta_recog(self,ego_traj, scene_name, nusc):
        """
        :param input_data: json格式的input
        :param nusc: Nuscene.nuscene.NuScene类
        :return: 根据ego vehicle, 判断input_data在地图场景中的位置，返回的是x坐标与y坐标的相对位移
        """
        
        normal_trace = ego_traj - ego_traj[0]
        ns_scene = nusc.get('scene', nusc.field2token('scene', 'name', scene_name)[0])
        sample_token = ns_scene['first_sample_token']
        sample = nusc.get('sample', sample_token)
        frame_id = 0
        trace_list = []
        while sample['next']:
            sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            annotation = nusc.get('ego_pose', sample_data['ego_pose_token'])
            trace_list.append(annotation['translation'][0:2])

            sample = nusc.get('sample', sample['next'])
            frame_id += 1
        trace_list = np.array(trace_list)

        frame_id = 0
        min_ade = 1e10
        min_frame = 100
        for i in range(len(trace_list)-len(normal_trace)+1):

            consider_trace = trace_list[i:i+len(normal_trace)]
            consider_trace = consider_trace - consider_trace[0]
            # ade = np.sqrt(np.sum((consider_trace-normal_trace)**2, axis=1)).mean()
            ade = np.linalg.norm(consider_trace - normal_trace, axis=1).mean()

            if ade <= min_ade:
                min_ade = ade.copy()
                min_frame = i
            frame_id += 1
        if min_frame == 100:
            print('no Frame matched!')
            return 0, 0
        delta = trace_list[min_frame] - ego_traj[0]
        delta_x = delta[0]
        delta_y = delta[1]
        # print('Consider Frame is:', min_frame,"min_ade",min_ade,delta_x,delta_y)
        return delta_x, delta_y

# 生成.index和.pkl 用于后续训练数据
if __name__ == '__main__':
    # for i in ['train', 'val', 'test']:
    for i in [ 'val' ]:
        # 创建一个数据集对象
        data_set = Dataset_Nuscene(
            flag=i,
            observe_len=4,
            pred_len=12,
            time_step=0.5,
            allow_incomplete_traces=True,
            allow_pedestrian_traces=True,
            allow_invisible_traces=False,
            load_mode='social') # 考虑social时，也需要考虑不完整的轨迹
        # 获取数据集的详细信息
        # print(i,data_set)