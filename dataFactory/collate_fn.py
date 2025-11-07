import numpy as np
import torch
from environment.map import GeometricMap    
import dill,os

def batch_norm(batch):
    """
    :param batch: list of dict
    :return:
    """    
    # 批量旋转地图
    scene_map = [a['vehicle_map'] for a in batch]
    map_point = torch.from_numpy(np.stack([a["agent_origin"] for a in batch], axis=0)).float()

    heading_angle = torch.tensor([a['rotate_angle'] for a in batch], dtype=torch.float)
    # patch_size = np.array([50, 10, 50, 90])   
    patch_size = np.array([44, 112, 180, 112]) 
    batch_agent_vehicle_map = GeometricMap.get_cropped_maps_from_scene_map_batch(scene_map, map_point, patch_size, rotation=heading_angle)
    # # batch_agent_vehicle_map_vit = GeometricMap.get_cropped_maps_from_scene_map_batch(scene_map, map_point, patch_for_vit, rotation=torch.full_like(heading_angle,90))

    # patch_for_vit = np.array([112, 44, 112, 180]) 
    # # patch_for_vit = np.array([50,50,50,50]) 
    # batch_agent_vehicle_map_vit = GeometricMap.get_cropped_maps_from_scene_map_batch(scene_map, map_point, patch_for_vit, rotation=heading_angle)


    # # 平移, batch_agent_pos, array of shape [BZ, TimeHorizon, 2]
    # batch_agent_pos = np.stack([(a['position'] - a['agent_origin']) for a in batch], axis=0)
    # time_horizon = batch_agent_pos.shape[1]
    # # 旋转
    # batch_rotate_mat = np.stack([a['rotate_mat'] for a in batch], axis=0)
    # batch_agent_pos = np.matmul(batch_agent_pos, batch_rotate_mat)

    # batch_observe_trace = [a['observe_trace'] for a in batch]
    # batch_future_trace = [a['future_trace'] for a in batch]
    # agent_data_list = [scene_map, map_point,batch_observe_trace,batch_future_trace,heading_angle,batch_agent_vehicle_map,batch_agent_vehicle_map_vit,batch_agent_pos]
    # data_dict_path = os.path.join('./precess_data/', 'nuScenes_batcrotatemap.pkl')
    # with open(data_dict_path, 'wb') as f:
    #     dill.dump(agent_data_list, f, protocol=dill.HIGHEST_PROTOCOL)
    # print("nuScenes_batcrotatemap.pkl saved!")
    # return
 
    # 批量判断类型
    has_motion_type = all('motion_type' in a for a in batch)
    if has_motion_type:
        batch_motion_type = [a['motion_type'] for a in batch]

    ################# 计算central agent位置
    # 平移, batch_agent_pos, array of shape [BZ, TimeHorizon, 2]
    batch_agent_pos = np.stack([(a['position'] - a['agent_origin']) for a in batch], axis=0)
    time_horizon = batch_agent_pos.shape[1]
    # 旋转
    batch_rotate_mat = np.stack([a['rotate_mat'] for a in batch], axis=0)
    batch_agent_pos = np.matmul(batch_agent_pos, batch_rotate_mat)# 转置旋转矩阵


    ################# 计算central agent 位移向量
    batch_agent_vec = np.zeros_like(batch_agent_pos)
    batch_agent_vec[:, 1:, :] = batch_agent_pos[:, 1:, :] - batch_agent_pos[:, :-1, :]

    ################# neighboring agent 相关
    batch_nbr_pos = []
    batch_nbr_rlpos_mask = []
    batch_nbr_rlpos = []
    batch_nbr_vec_mask = []
    batch_nbr_vec = []
    for idx, data in enumerate(batch):
        ################# neighboring agent = 0 时的情况
        if data['nbr_agent_position'].shape[0] == 0:
            nbr_pos = np.zeros((100, time_horizon, 2))
            nbr_rlpos_mask = np.zeros((100, time_horizon))
            nbr_rlpos = np.zeros((100, time_horizon, 2))
            nbr_vec_mask = np.zeros((100, time_horizon))
            nbr_vec = np.zeros((100, time_horizon, 2))
        else:
            ################# nbr 位置
            nbr_num, _, _ = data['nbr_agent_position'].shape
            # nbr_agent_position: [MaxAgentNum, TimeHorizon 2-xy],
            if nbr_num > 100:
                raise Exception('Neighboring agents exceeds maximum number of agents, 100')
            nbr_pos = np.zeros((100, time_horizon, 2))
            # nbr 位移
            nbr_pos[:nbr_num, :, :] = (data['nbr_agent_position'] - data['agent_origin']) * \
                                      np.repeat(np.expand_dims(data['nbr_agent_mask'], axis=-1), 2, axis=-1)
            # nbr 旋转：
            nbr_pos = np.matmul(nbr_pos, data['rotate_mat'])
            # ps: center agent 经历了位移旋转，neighboring agents 也经历了位移旋转，那么后面的相对位置和位移向量不用再位移旋转了

            ################# nbr 相对位置
            # nbr_rlpos_mask: 相对位置mask, [MaxNbrAgentNum, TimeHorizon]
            nbr_rlpos_mask = np.zeros((100, time_horizon))
            nbr_rlpos_mask[:nbr_num, :] = data['nbr_agent_mask']
            # 计算相对位置
            nbr_rlpos = (nbr_pos - batch_agent_pos[idx])
            # 相对位置x相对位置mask
            nbr_rlpos = nbr_rlpos * np.repeat(np.expand_dims(nbr_rlpos_mask, axis=-1), 2, axis=-1)

            ################# nbr 位移向量
            # 获取nbr 位移向量的mask
            nbr_vec_mask = np.zeros_like(nbr_rlpos_mask)
            # 错位相乘，连续时刻均不为0才为1
            nbr_vec_mask[:, 1:] = nbr_rlpos_mask[:, :-1] * nbr_rlpos_mask[:, 1:]
            # 计算位移向量
            nbr_vec = np.zeros_like(nbr_pos)
            nbr_vec[:, 1:, :] = nbr_pos[:, 1:, :] - nbr_pos[:, :-1, :]
            # 位移向量x位移向量mask
            nbr_vec = nbr_vec * np.repeat(np.expand_dims(nbr_vec_mask, axis=-1), 2, axis=-1)

        batch_nbr_pos.append(nbr_pos)
        batch_nbr_rlpos_mask.append(nbr_rlpos_mask)
        batch_nbr_rlpos.append(nbr_rlpos)
        batch_nbr_vec_mask.append(nbr_vec_mask)
        batch_nbr_vec.append(nbr_vec)

    batch_nbr_pos = np.stack(batch_nbr_pos, axis=0)
    batch_nbr_rlpos_mask = np.stack(batch_nbr_rlpos_mask, axis=0)
    batch_nbr_rlpos = np.stack(batch_nbr_rlpos, axis=0)
    batch_nbr_vec_mask = np.stack(batch_nbr_vec_mask, axis=0)
    batch_nbr_vec = np.stack(batch_nbr_vec, axis=0)
    if has_motion_type:
        return {'batch_agent_pos': batch_agent_pos,  # array of shape (BZ, TimeHorizon, 2), 一批 central agent的xy坐标
                'batch_rotate_mat': batch_rotate_mat,  # array of shape (BZ, 2，2)，由当前时刻central agent 的 vec所确定的旋转矩阵
                'batch_agent_vec': batch_agent_vec,  # array of shape (BZ, TimeHorizon, 2), 一批 central agent的vec
                'batch_nbr_pos': batch_nbr_pos,
                # array of shape (BZ, MaxAgentNum (=100)，TimeHorizon, 2), neighboring agent 的xy坐标
                'batch_nbr_rlpos_mask': batch_nbr_rlpos_mask,
                'batch_nbr_rlpos': batch_nbr_rlpos,
                'batch_nbr_vec_mask': batch_nbr_vec_mask,
                'batch_nbr_vec': batch_nbr_vec,
                'batch_agent_vehicle_map':batch_agent_vehicle_map,
                'batch_motion_type':batch_motion_type,
                },has_motion_type
    else:
        return {'batch_agent_pos': batch_agent_pos,  # array of shape (BZ, TimeHorizon, 2), 一批 central agent的xy坐标
                        'batch_rotate_mat': batch_rotate_mat,  # array of shape (BZ, 2，2)，由当前时刻central agent 的 vec所确定的旋转矩阵
                        'batch_agent_vec': batch_agent_vec,  # array of shape (BZ, TimeHorizon, 2), 一批 central agent的vec
                        'batch_nbr_pos': batch_nbr_pos,
                        # array of shape (BZ, MaxAgentNum (=100)，TimeHorizon, 2), neighboring agent 的xy坐标
                        'batch_nbr_rlpos_mask': batch_nbr_rlpos_mask,
                        'batch_nbr_rlpos': batch_nbr_rlpos,
                        'batch_nbr_vec_mask': batch_nbr_vec_mask,
                        'batch_nbr_vec': batch_nbr_vec,
                        'batch_agent_vehicle_map':batch_agent_vehicle_map,
                        },has_motion_type
def array2tensor(dict_data, has_motion_type):
    batch_agent_pos = torch.from_numpy(dict_data['batch_agent_pos']).to(torch.float32)
    # batch_rotate_mat = torch.from_numpy(dict_data['batch_rotate_mat']).to(torch.float32)
    batch_agent_vec = torch.from_numpy(dict_data['batch_agent_vec']).to(torch.float32)
    # neighboring agents
    # batch_nbr_pos = torch.from_numpy(dict_data['batch_nbr_pos']).to(torch.float32)
    batch_nbr_rlpos_mask = torch.from_numpy(dict_data['batch_nbr_rlpos_mask']).to(torch.float32)
    batch_nbr_rlpos = torch.from_numpy(dict_data['batch_nbr_rlpos']).to(torch.float32)
    batch_nbr_vec_mask = torch.from_numpy(dict_data['batch_nbr_vec_mask']).to(torch.float32)
    batch_nbr_vec = torch.from_numpy(dict_data['batch_nbr_vec']).to(torch.float32)
    
    # For 'batch_agent_vehicle_map', handle it if it's already a tensor:
    batch_agent_vehicle_map = dict_data['batch_agent_vehicle_map']
    if isinstance(batch_agent_vehicle_map, np.ndarray):  # If it's a numpy array, convert it to a tensor
        batch_agent_vehicle_map = torch.from_numpy(batch_agent_vehicle_map)
    batch_agent_vehicle_map = batch_agent_vehicle_map.to(torch.float32)
    if has_motion_type:
        batch_motion_type = batch_motion_type.to(torch.float32)
        return batch_agent_vec, batch_nbr_rlpos_mask, batch_nbr_rlpos, batch_nbr_vec_mask, batch_nbr_vec, batch_agent_pos, batch_agent_vehicle_map, batch_motion_type
    
    return batch_agent_vec, batch_nbr_rlpos_mask, batch_nbr_rlpos, batch_nbr_vec_mask, batch_nbr_vec, batch_agent_pos, batch_agent_vehicle_map

def collate_fn(batch):
    normed_batch,has_motion_type = batch_norm(batch)
    return  array2tensor(normed_batch,has_motion_type)