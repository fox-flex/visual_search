import os
import os.path as osp
from pathlib import Path
from glob import glob
import shutil
import argparse

import json
import pickle

import subprocess

from tqdm.auto import tqdm
import cv2

class GenDataset:
    @staticmethod
    def gen_dataset(in_data_path: str, datasets_dir: Path, dataset_name: str, force_copy_imgs: bool = False, gen_superglobal: bool = False, gen_ames: bool = False):
        query = 'query' in dataset_name
        GenDataset._gen_ds(in_data_path, datasets_dir, dataset_name, query=query, force_copy_imgs=force_copy_imgs)
        if gen_superglobal:
            GenDataset.preprocess_superglobal(datasets_dir, dataset_name)
        if gen_ames:
            GenDataset.preprocess_ames(datasets_dir, dataset_name)

    @staticmethod
    def _gen_ds(in_data_path: str, datasets_dir: Path, dataset_name: str, query: bool, force_copy_imgs: bool):
        output_dir = datasets_dir / dataset_name

        output_dir.mkdir(exist_ok=True, parents=True)
        jpg_dir = output_dir / 'jpg'

        files = list(os.listdir(in_data_path))
        jpg_files = [file_name
                     for file_name in files
                     if file_name.endswith('.jpg')]
        if query:
            gnd = []
        
        split_string = 'query' if query else 'gallery'
        names_txt = str(output_dir / f'{dataset_name}_{split_string}.txt')
        
        with open(names_txt, 'w') as f:
            f.write('\n'.join(jpg_files))

        if not force_copy_imgs and len(files) == len(jpg_files) and not query:
            os.symlink(os.path.abspath(in_data_path), jpg_dir)
        else:
            os.makedirs(jpg_dir, exist_ok=True)
            for file_path in tqdm(jpg_files):
                img_path = os.path.join(in_data_path, file_path)
                shutil.copy2(img_path, jpg_dir)

                if query:
                    size = cv2.imread(img_path).shape[:2][::-1]
                    gnd.append({
                        'bbx': [0, 0, *size]
                    })


        names = [file.split('.')[0] for file in jpg_files]
        if query:
            data = {
                'qimlist': names,
                'gnd': gnd
            }
        else:
            data = {
                'imlist': names
            }
            
        with open(os.path.join(output_dir, f'gnd_{dataset_name}.pkl'), 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def preprocess_superglobal(datasets_dir: Path, dataset_name: str):
        cwd = 'superglobal'
        reverse_cwd = '..'
        data_dir = osp.join(reverse_cwd, datasets_dir)
        args = [
            "MODEL.DEPTH", "50",
            "TEST.WEIGHTS", "./weights/CVPR2022_CVNet_R50.pyth",
            "TEST.DATA_DIR", data_dir,
            "TEST.DATASET_LIST", f'[["{dataset_name}", ""]]'
        ]
        subprocess.run(["python", "test.py"] + args, cwd=cwd)
    
    def preprocess_ames(datasets_dir: Path, dataset_name: str):
        cwd = 'ames'
        reverse_cwd = '..'
        data_dir = osp.join(reverse_cwd, datasets_dir)
        split = '_query' if 'query' in dataset_name else '_gallery'
        args = [
            "--dataset", dataset_name, 
            "--split", split,
            "--backbone", "cvnet",
            "--weight", "chpt/CVPR2022_CVNet_R50.pyth",
            "--save_path", data_dir,
            "--data_path", data_dir,
            "--desc_type", "local",
            "--detector", "chpt/cvnet_detector.pt"
        ]

        extra_path = osp.join(osp.dirname(osp.abspath(__file__)), cwd)
        python_path = os.environ.get('', '')
        os.environ['PYTHONPATH'] = f'{extra_path}:{python_path}'

        subprocess.run(['python', 'extract/extract_descriptors.py'] + args, cwd=cwd)

    
def get_args():
    parser = argparse.ArgumentParser(description='Merge hdf5 files.')
    parser.add_argument('--in_data_path',
                        help='Path to dir with images.')
    parser.add_argument('--dataset_name',
                        type=str,
                        help='Name of created dataset.')
    parser.add_argument('--dataset_dir',
                        type=str,
                        default='data/datasets',
                        help='Generated datasets directory.')
    parser.add_argument('--force_copy_imgs',
                        action='store_true',
                        help='If the images are in valid form the symlink to the original directory would be created. ' \
                             'If --force_copy_imgs, the images would be forcly copied.')
    parser.add_argument('--gen_superglobal',
                        action='store_true',
                        help='Gen superglobal descriptors.')
    parser.add_argument('--gen_ames',
                        action='store_true',
                        help='Gen ames descriptors.')
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = get_args()
    in_data_path = args.in_data_path
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    
    datasets_dir = Path(dataset_dir)
    GenDataset.gen_dataset(in_data_path, datasets_dir, dataset_name,
                          gen_superglobal=False,
                          gen_ames=args.gen_ames)

