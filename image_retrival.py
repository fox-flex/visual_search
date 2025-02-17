import os
import os.path as osp
import json
from pathlib import Path
import subprocess

from image_roi_detector import ImageROIDetector
from gen_dataset import GenDataset

class ImageRetrival:
    def __init__(
        self,
        data_dir: str = './data',
        gdino_tocken_path: str = ''
    ):
        self.data_dir = Path(data_dir)
        self.datasets_dir = self.data_dir / 'datasets'
        self.rois_path = str(self.data_dir / 'rois')
        self.query_ds_name = 'query'
        self.image_roi_detector = ImageROIDetector(token_path=gdino_tocken_path)
    
    def build_dataset(self, imgs_dir: str, dataset_name: str = 'ds'):
        if (self.datasets_dir / dataset_name).exists():
            print(f'dataset "{dataset_name}" already exist')
            return
        print(f'dataset "{dataset_name}" does not exist, generating it.')
        GenDataset.gen_dataset(imgs_dir, self.datasets_dir, dataset_name)

    def build_query(self, img_path: str, query_prompt: str = '', gen_superglobal: bool = False, gen_ames: bool = False):
        self.image_roi_detector.gen_rois(img_path, self.rois_path, query_prompt)
        GenDataset.gen_dataset(self.rois_path, self.datasets_dir, self.query_ds_name,
                               gen_superglobal=gen_superglobal, gen_ames=gen_ames)
    
    def infer_superglobal(self, dataset_name: str, imgs_per_crop: int) -> list[str]:
        cwd = 'superglobal'
        reverse_cwd = '..'
        data_dir = osp.join(reverse_cwd, self.datasets_dir)
        args = [
            "IMGS_PER_QUERY", f'{imgs_per_crop}',
            "MODEL.DEPTH", "50",
            "TEST.WEIGHTS", "./weights/CVPR2022_CVNet_R50.pyth",
            "TEST.DATA_DIR", data_dir,
            "TEST.DATASET_LIST", f'[["{dataset_name}", "{self.query_ds_name}"]]'
        ]

        subprocess.run(["python", "test.py"] + args, cwd=cwd)
        out_file = osp.join(self.datasets_dir, 'sim.json')
        with open(out_file, "r") as f:
            data = json.load(f)
        img_paths = {
            osp.join(cwd, img_q): [osp.join(cwd, sim_img) for sim_img in sim_imgs]
            for img_q, sim_imgs in data.items()
        }
        return img_paths
    
    def infer_ames(self, dataset_name: str, imgs_per_crop: int) -> list[str]:
        dataset_desc_path = osp.join(self.datasets_dir, dataset_name, 'cvnet_gallery_local.hdf5')
        if not osp.exists(dataset_desc_path):
            print('The ames features not detected. Starting preprocess')
            GenDataset.preprocess_ames(self.datasets_dir, dataset_name)
        GenDataset.preprocess_ames(self.datasets_dir, self.query_ds_name)
        cwd = 'ames'
        reverse_cwd = '..'
        data_dir = osp.join(reverse_cwd, self.datasets_dir)

        args = [
            "--multirun",
            "desc_name=cvnet",
            f"data_root={data_dir}",
            f"imgs_per_query={imgs_per_crop}",
            "resume=./chpt/r101_cvnet_ames.pt",
            "model.binarized=False",
            f"dataset@test_dataset_db={dataset_name}",
            f"dataset@test_dataset_q={self.query_ds_name}",
            "test_dataset.query_sequence_len=600",
            "test_dataset.sequence_len=50",
            "test_dataset.batch_size=1",
            "test_dataset.lamb=[0.55]",
            "test_dataset.temp=[0.3]",
            "test_dataset.num_rerank=[100]"
        ]

        extra_path = osp.join(osp.dirname(osp.abspath(__file__)), cwd)
        python_path = os.environ.get('', '')
        os.environ['PYTHONPATH'] = f'{extra_path}:{python_path}'
        
        subprocess.run(['python', 'src/evaluate.py'] + args, cwd=cwd)
        
        out_file = os.path.join(self.datasets_dir, 'sim.json')
        with open(out_file, "r") as f:
            data = json.load(f)
        img_paths = {
            os.path.join(cwd, img_q): [os.path.join(cwd, sim_img) for sim_img in sim_imgs]
            for img_q, sim_imgs in data.items()
        }
        return img_paths
    
    def infer(self, img_path: str, query_prompt: str = '', dataset_name: str = 'ds', imgs_per_crop: int = 5, detection_mode: str = 'superglobal'):
        dataset_dir = osp.join(self.datasets_dir, dataset_name)
        if not osp.exists(dataset_dir):
            raise RuntimeError(f'No dataset by path: {dataset_dir}')

        self.build_query(img_path, query_prompt)
        if detection_mode == 'superglobal':
            res = self.infer_superglobal(dataset_name, imgs_per_crop)
        elif detection_mode == 'ames':
            res = self.infer_ames(dataset_name, imgs_per_crop)
        else:
            raise NotImplementedError(f'Unknown {detection_mode=}')

        return res
        

        


if __name__ == '__main__':
    # retrival = ImageRetrival(gdino_tocken_path=None)
    retrival = ImageRetrival()
    img_path = './data/datasets/my_test/jpg/9b656a1e0593347e.jpg'
    # imgs_dir = 'data/test_data'
    # dataset_name = 'ds_s'
    imgs_dir = 'data/test_task_data'
    dataset_name = 'ds'
    
    retrival.build_dataset(imgs_dir, dataset_name)
    # retrival.build_query(img_path)
    # retrival.infer_superglobal(dataset_name)
