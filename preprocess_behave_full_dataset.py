#bsub -n 1 -W 24:00 -o log_preprocess_full_dataset -R "rusage[mem=2G]"  python preprocess_behave_full_dataset.py
import os
import pdb
import cv2
import json
import time
import trimesh
from plyfile import PlyData, PlyElement

import numpy as np
import matplotlib.pyplot as plt
from data.frame_data import FrameDataReader
from data.kinect_transform import KinectTransform
from data.pc_utils import save_point_cloud, save_point_cloud_w_segm

CLASS_LABELS = ('background', 'backpack', 'boxmedium', 'chairwood', 'stool', 'toolbox', 
                'basketball', 'boxsmall', 'keyboard', 'suitcase', 'trashbin', 
                'boxlarge', 'boxtiny', 'monitor', 'tablesmall', 'yogaball', 
                'boxlong', 'chairblack', 'plasticcontainer', 'tablesquare', 'yogamat', 'person')

LABEL2ID_DICT = {val:id for id, val in enumerate(CLASS_LABELS)}

BEHAVE_COLOR_MAP = {
    0: (0., 0., 0.), #background
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    13: (82., 84., 163.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (100., 85., 144.), 
    21: (149., 0., 58.) #person
}

dataset_root = '/cluster/project/infk/263-5906-00L/data/BEHAVE'
sequences_root = os.path.join(dataset_root, 'sequences')
print('[INFO] Original dataset root:', dataset_root)

w, h = 1536, 2048
SAMPLE_KEEP_RATE = 0.20 # for downsampling the point cloud, we will keep SAMPLE_KEEP_RATE percent of the points for each scene
seed = 42
np.random.seed(seed)
print('[INFO] w:', w, 'h:', h)
print('[INFO] SAMPLE_KEEP_RATE:', SAMPLE_KEEP_RATE)
print('[INFO] np.random.seed(' + str(seed) + ')')

splits = ['train', 'val', 'test']
#splits = ['val']
#splits = ['test']

#split_dict_path = '/cluster/project/infk/263-5906-00L/data/BEHAVE/split.json'
split_dict_path = 'split_trainvaltest.json'
processed_data_out_dir = '/cluster/scratch/takmaza/VirtualHumans/behave_preprocessed'
print('[INFO] Processed data will be saved to:', processed_data_out_dir)

if not os.path.exists(processed_data_out_dir):
    os.makedirs(processed_data_out_dir)

processed_data_split_dir_dict = {split:os.path.join(processed_data_out_dir, split) for split in splits}
for key, val in processed_data_split_dir_dict.items():
    if not os.path.exists(val):
        os.makedirs(val)

with open(split_dict_path, 'r') as file:
    split_dict = json.load(file)

if 'val' not in split_dict.keys():
    print('[INFO] Split dict does not have the val split! Working on the issue...')
    np.random.seed(42)
    train_paths = split_dict['train'] #231 sequences -> 0.72
    test_paths = split_dict['test'] #90 sequences -> 0.28
    num_val_seqs = int((len(train_paths)+len(test_paths))*0.05) #16
    val_inds = np.random.choice(range(len(train_paths)), num_val_seqs)
    train_inds = np.setdiff1d(range(len(train_paths)), val_inds)
    new_train_paths = [train_paths[id] for id in train_inds] #216 sequences -> 0.67
    new_val_paths = [train_paths[id] for id in val_inds] # 16 sequences -> 0.05
    
    split_write_root = 'split_trainvaltest.json'
    if os.path.exists(split_write_root):
        raise Exception('Train-val-test split already exists for the Behave dataset. Please specify ' + str(split_write_root) + ' as the split file in your config!')
    else:
        with open(split_write_root, 'w') as file:
            new_split_dict = {'train':new_train_paths, 'val':new_val_paths, 'test':test_paths}
            json.dump(new_split_dict, file, indent=2)
            print('[INFO] Saved new split dict with train/val/test to ' + split_write_root)
    split_dict = new_split_dict
else:
    print('[INFO] Successfully loaded the train/val/test split from ' + split_dict_path)

preprocessed_dataset_split_backup_path = os.path.join(processed_data_out_dir, 'split_trainvaltest.json')
with open(preprocessed_dataset_split_backup_path, 'w') as file:
    json.dump(split_dict, file, indent=2)
    print('[INFO] Created a backup of the split dict with train/val/test at ' + preprocessed_dataset_split_backup_path)

overall_time_start = time.time()
for split in splits: #['train', 'val', 'test']
    current_split = split_dict[split]
    current_split_out_dir = processed_data_split_dir_dict[split]
    counter = 0
    for seq_name in current_split: #'Date01_Sub01_backpack_back'
        current_seq_out_dir = os.path.join(current_split_out_dir, seq_name)
        seq_path = os.path.join(sequences_root, seq_name)
        reader = FrameDataReader(seq_path)
        kinect_transform = KinectTransform(seq_path, kinect_count=reader.kinect_count)

        seq_end = reader.cvt_end(None)
        loop = range(0, seq_end)
        interaction_obj_type = reader.seq_info.get_obj_name()

        for id in loop:
            # get all color images in this frame
            kids = [0, 1, 2, 3] # choose which kinect id to visualize
            imgs_all = reader.get_color_images(id, reader.kids)
            depths_all = reader.get_depth_images(id, reader.kids) #depth values are in milimeters

            imgs_resize = [cv2.resize(x, (w, h)) for x in imgs_all]
            depths_resize = [cv2.resize(x, (w, h)) for x in depths_all]

            frame_folder_id = reader.get_frame_folder(id).split('/')[-1]
            current_frame_folder_out_dir = os.path.join(current_seq_out_dir, frame_folder_id)
            if not os.path.exists(current_frame_folder_out_dir):
                os.makedirs(current_frame_folder_out_dir)

            for kid, rgb, dpt in zip(kids, imgs_all, depths_all):
                current_frame_out_ply_path = os.path.join(current_frame_folder_out_dir, 'k_'+str(kid)+'.ply')
                if os.path.exists(current_frame_out_ply_path):
                    print('[INFO] Skipped ' + current_frame_out_ply_path + ' - already exists!')
                    counter += 1
                    continue
                start=time.time()
                person_mask = reader.get_mask(id, kid, 'person', ret_bool=True)
                obj_mask = reader.get_mask(id, kid, 'obj', ret_bool=True)
                if obj_mask is None:
                    print('[INFO] Skipped ' + current_frame_out_ply_path + ' - there is no object mask!')
                    counter += 1
                    continue
                
                pc_filtered, valid_mask, pc = kinect_transform.dmap2pc_shaped(dpt, kid)
                person_obj_no_intersection_mask = (1-obj_mask*person_mask) #(1536, 2048)
                updated_valid_mask = np.array(valid_mask * person_obj_no_intersection_mask, dtype=bool) #(1536, 2048) -> (1536, 2048)*(1536, 2048)
                updated_valid_mask_3D = np.array(np.repeat(np.expand_dims(updated_valid_mask,axis=2), 3, axis=2), dtype=bool)
                
                label_map = np.zeros(dpt.shape)
                label_map[obj_mask==1]=LABEL2ID_DICT[interaction_obj_type]
                label_map[person_mask==1]=LABEL2ID_DICT['person']

                segm_color_map = np.zeros(rgb.shape)
                segm_color_map[obj_mask==1]=BEHAVE_COLOR_MAP[LABEL2ID_DICT[interaction_obj_type]]
                segm_color_map[person_mask==1]=BEHAVE_COLOR_MAP[LABEL2ID_DICT['person']]
                
                # sampling points
                num_points = updated_valid_mask.sum()
                num_points_low_res = int(num_points * SAMPLE_KEEP_RATE)
                idx_points_low_res = np.random.choice(updated_valid_mask.sum(), num_points_low_res)
                
                # taking the sampled subset of the masked input
                pc_out = pc[updated_valid_mask][idx_points_low_res, :]
                rgb_out = rgb[updated_valid_mask,:][idx_points_low_res, :]
                segm_rgb_out = segm_color_map[updated_valid_mask][idx_points_low_res, :]
                label_out = label_map[updated_valid_mask][idx_points_low_res]
                
                # saving the ply
                points_3d = np.zeros((pc_out.shape[0], 10))
                points_3d[:,0:3] = pc_out
                points_3d[:,3:6] = rgb_out
                points_3d[:,6:9] = segm_rgb_out
                points_3d[:,9] = label_out

                ply_path_verbose_bool = True if counter%100==0 else False
                save_point_cloud_w_segm(points_3d, current_frame_out_ply_path, binary=True, verbose=ply_path_verbose_bool)
                end=time.time()
                if ply_path_verbose_bool: 
                    print('[INFO] Processed the frame in ' + str((end-start)) + ' seconds.')
                counter += 1
        print('[INFO] Sequence ' + seq_name + ' processed. Total frame count since the split started: ', str(counter))
    print('[INFO] Split ' + split + ' processed. Total frame count in the split:',  str(counter))
overall_time_end = time.time()
print('[INFO] Processing complete! Processed files can be found in', processed_data_out_dir)
overall_time = overall_time_end-overall_time_start
overall_min = int(overall_time//60)
overall_rem_sec = int(overall_time%60)
print('[INFO] Processing took', str(overall_min) ,'minutes', str(overall_rem_sec), 'seconds.')
