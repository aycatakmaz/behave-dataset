import os
import pdb
import cv2
import json
import numpy as np
from data.frame_data import FrameDataReader
from data.kinect_transform import KinectTransform

dataset_root = '/cluster/project/infk/263-5906-00L/data/BEHAVE'
calib_root = os.path.join(dataset_root, 'calibs')
objects_root = os.path.join(dataset_root, 'objects')
sequences_root = os.path.join(dataset_root, 'sequences')

image_size = 640
w, h = image_size, int(image_size * 0.75)

splits = ['train', 'val', 'test']
#split_dict_path = '/cluster/project/infk/263-5906-00L/data/BEHAVE/split.json'
split_dict_path = 'split_trainvaltest.json'
processed_data_out_dir = '/cluster/scratch/takmaza/VirtualHumans/behave_preprocessed'

if not os.path.exists(processed_data_out_dir):
    os.makedirs(processed_data_out_dir)

processed_data_split_dir_dict = {split:os.path.join(processed_data_out_dir, split) for split in splits}
for key, val in processed_data_split_dir_dict.items():
    if not os.path.exists(val):
        os.makedirs(val)

with open(split_dict_path, 'r') as file:
    split_dict = json.load(file)
if 'val' not in split_dict.keys():
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
            print('Saved new split dict with train/val/test to ' + split_write_root)

for split in splits: #['train', 'val', 'test']
    current_split = split_dict[split]
    current_split_out_dir = processed_data_split_dir_dict[split]
    for seq_name in current_split: #'Date01_Sub01_backpack_back'
        seq_path = os.path.join(sequences_root, seq_name)
        #seq_name_parts = seq_name.split('_') #['Date01', 'Sub01', 'backpack', 'back']
        #date_name, sub_name, action_name = seq_name_parts[0], seq_name_parts[1], '_'.join(seq_name_parts[2:]) #'Date01', 'Sub01', 'backpack_back'
        #current_calib_dir = os.path.join(calib_root, date_name)

        reader = FrameDataReader(seq_path)
        kinect_transform = KinectTransform(seq_path, kinect_count=reader.kinect_count)
        pdb.set_trace()

        seq_end = reader.cvt_end(None)
        loop = range(0, seq_end)
   
        for id in loop:
            # get all color images in this frame
            kids = [0, 1, 2, 3] # choose which kinect id to visualize
            imgs_all = reader.get_color_images(id, reader.kids)
            depths_all = reader.get_depth_images(id, reader.kids)

            imgs_resize = [cv2.resize(x, (w, h)) for x in imgs_all]
            depths_resize = [cv2.resize(x, (w, h)) for x in depths_all]

            selected_imgs = [imgs_resize[x] for x in kids] # here we render fitting in all 4 views
            for orig, kid in zip(selected_imgs, kids):
                h, w = orig.shape[:2]
            
            # load person and object mask
            res = []
            for kid, rgb in zip(kids, imgs_all):
                obj_mask = np.zeros_like(rgb).astype(np.uint8)
                mask = reader.get_mask(i, kid, 'obj', ret_bool=True)
                if mask is None:
                    continue # mask can be None if there is not fitting in this frame
                obj_mask[mask] = np.array([255, 0, 0])
            
                person_mask = np.zeros_like(rgb).astype(np.uint8)
                mask = reader.get_mask(i, kid, 'person', ret_bool=True)
                person_mask[mask] = np.array([255, 0, 0])
            
                comb = np.concatenate([rgb, person_mask, obj_mask], 1)
                ch, cw = comb.shape[:2]
                res.append(cv2.resize(comb, (cw//3, ch//3)))
            

            # load person and object pc, return psbody.Mesh
            # convert flag is used to be compatible with detectron2 classes, in detectron2 all chairs are clasified as chair,
            # so the chair pc is saved in subfolder chair; also all yogaball, basketball are classified as 'sports ball',
            # obj_pc = reader.get_pc(i, 'obj', convert=True)
            # person_pc = reader.get_pc(i, 'person')

            # load person and object mask
            # for kid, rgb, writer in zip(kids, imgs_all, video_writers):
            #     obj_mask = np.zeros_like(rgb).astype(np.uint8)
            #     mask = reader.get_mask(i, kid, 'obj', ret_bool=True)
            #     if mask is None:
            #         continue # mask can be None if there is not fitting in this frame
            #     obj_mask[mask] = np.array([255, 0, 0])
            #
            #     person_mask = np.zeros_like(rgb).astype(np.uint8)
            #     mask = reader.get_mask(i, kid, 'person', ret_bool=True)
            #     person_mask[mask] = np.array([255, 0, 0])
            #
            #     comb = np.concatenate([rgb, person_mask, obj_mask], 1)
            #     ch, cw = comb.shape[:2]
            #     writer.append_data(cv2.resize(comb, (cw//3, ch//3)))
