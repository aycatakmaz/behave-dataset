import os
import sys
import torch
import logging
from pathlib import Path
import numpy as np
import plyfile

import pdb
import json
from utils import read_txt

CLASS_LABELS = ('backpack', 'boxmedium', 'chairwood', 'stool', 'toolbox', 
                'basketball', 'boxsmall', 'keyboard', 'suitcase', 'trashbin', 
                'boxlarge', 'boxtiny', 'monitor', 'tablesmall', 'yogaball', 
                'boxlong', 'chairblack', 'plasticcontainer', 'tablesquare', 'yogamat', 'person', 'others')
VALID_CLASS_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)

BEHAVE_COLOR_MAP = {
    0: (0., 0., 0.),
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
    20: (149., 0., 58.), #person
    21: (100., 85., 144.), #others
}

from enum import Enum
class DatasetPhase(Enum):
  Train = 0
  Val = 1
  Val2 = 2
  TrainVal = 3
  Test = 4

class BehaveSparseVoxelizationDataset():

    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.05

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                        np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
    # Original SCN uses
    # ELASTIC_DISTORT_PARAMS = ((2, 4), (8, 8))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    NUM_LABELS = len(CLASS_LABELS)  # Will be converted to 20 as defined in IGNORE_LABELS.
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))
    IS_FULL_POINTCLOUD_EVAL = True

    def __init__(self,
               config,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               elastic_distortion=False,
               cache=False,
               phase=DatasetPhase.Train):
        if isinstance(phase, str):
            phase = self.str2datasetphase_type(phase)
        # Use cropped rooms for train/val
        data_root = config.behave_path
        split_root = config.behave_split_path

        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        
        with open(split_root, 'r') as file:
            split_dict = json.load(file)

        if 'val' in split_dict.keys():
            self.train_paths = split_dict['train']
            self.val_paths = split_dict['val']
            self.trainval_paths = sorted(self.train_paths + self.val_paths)
            self.test_paths = split_dict['test']

        else:
            np.random.seed(42)
            train_paths = split_dict['train'] #231 sequences -> 0.72
            test_paths = split_dict['test'] #90 sequences -> 0.28
            num_val_seqs = int((len(train_paths)+len(test_paths))*0.05) #16
            val_inds = np.random.choice(range(len(train_paths)), num_val_seqs)
            train_inds = np.setdiff1d(range(len(train_paths)), val_inds)
            new_train_paths = [train_paths[id] for id in train_inds] #216 sequences -> 0.67
            new_val_paths = [train_paths[id] for id in val_inds] # 16 sequences -> 0.05
            
            split_write_root = ''.join(split_root.split('.')[:-1]) + '_trainvaltest.json'
            if os.path.exists(split_write_root):
                raise Exception('Train-val-test split already exists for the Behave dataset. Please specify ' + str(split_write_root) + ' as the split file in your config!')
            else:
                with open(split_write_root, 'w') as file:
                    new_split_dict = {'train':new_train_paths, 'val':new_val_paths, 'test':test_paths}
                    json.dump(new_split_dict, file, indent=2)

            self.train_paths = new_train_paths
            self.val_paths = new_val_paths
            self.trainval_paths = sorted(train_paths)
            self.test_paths = test_paths
        
        if phase == DatasetPhase.Train:
            self.data_paths_orig = self.train_paths
        elif phase == DatasetPhase.Val:
            self.data_paths_orig = self.val_paths
        elif phase == DatasetPhase.TrainVal:
            self.data_paths_orig = self.trainval_paths
        else:
            self.data_paths_orig = self.test_paths

        self.data_paths = [os.path.join(data_root, el) for el in self.data_paths_orig]

        #pdb.set_trace()
        logging.info('Loading {}: {}'.format(self.__class__.__name__, phase))
        print('Loading {}: {}'.format(self.__class__.__name__, phase))
        '''
        super().__init__(
            data_paths,
            data_root=data_root,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=config.ignore_label,
            return_transformation=config.return_transformation,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config)
        '''

    def get_output_id(self, iteration):
        return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

    def str2datasetphase_type(self, arg):
        if arg.upper() == 'TRAIN':
            return DatasetPhase.Train
        elif arg.upper() == 'VAL':
            return DatasetPhase.Val
        elif arg.upper() == 'VAL2':
            return DatasetPhase.Val2
        elif arg.upper() == 'TRAINVAL':
            return DatasetPhase.TrainVal
        elif arg.upper() == 'TEST':
            return DatasetPhase.Test
        else:
            raise ValueError('phase must be one of train/val/test')

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self):
        return None


class Struct:
    def __init__(self):
        return

    def update_struct(self, **entries):
        self.__dict__.update(entries)

if __name__ == '__main__':
    config=Struct()
    config.update_struct(**{
        'behave_path':'/Users/aycatakmaz/Projects/behave-dataset-dataloader/BEHAVE',
        'behave_split_path':'/Users/aycatakmaz/Projects/behave-dataset-dataloader/BEHAVE/split_trainvaltest.json'
        })

    BehaveSparseVoxelizationDataset(config,
        input_transform=None,
        target_transform=None,
        augment_data=True,
        elastic_distortion=False,
        cache=False,
        phase=DatasetPhase.Train)

    
