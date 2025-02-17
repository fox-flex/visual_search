import argparse
import os
import shutil
from pathlib import Path
from gdino import GroundingDINOAPIWrapper, visualize
from PIL import Image
import numpy as np
import cv2


class ImageROIDetector:
    def __init__(self, token_path: str|None = None):
        try:
            with open(token_path, 'r') as f:
                token = f.read().strip()
            self.gdino = GroundingDINOAPIWrapper(token)
        except Exception as e:
            self.gdino = None
            print(f'Warning: cant load GroundingDINOAPI')
        
        self.prompts_str = 'main object.key object.logo'

    
    def gen_rois(self, img_path: str, out_data_dir: str, query_prompt: str = ''):
        out_data_dir = Path(out_data_dir)
        if out_data_dir.exists():
            shutil.rmtree(str(out_data_dir))
        out_data_dir.mkdir(parents=True)
        img = cv2.imread(img_path)
        cv2.imwrite(str(out_data_dir / 'img.jpg'), img)
        
        if self.gdino is None:
            return
        
        text_prompt = self.prompts_str
        if query_prompt:
            text_prompt = f'{query_prompt}.{text_prompt}'
        print(f'{text_prompt=}')
        prompts = {
            'image': img_path,
            'prompt': text_prompt
        }
        results = self.gdino.inference(prompts)
        for i, bbox in enumerate(results['boxes']):
            x0, y0, x1, y1 = bbox
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        
            img_bbox = img[y0:y1, x0:x1]
            file_path = str(out_data_dir / f'img_{i}.jpg')
            cv2.imwrite(file_path, img_bbox)

        
if __name__ == "__main__":
    img_path = './data/datasets/my_test/jpg/9b656a1e0593347e.jpg'
    out_data_dir = './data/rois'
    roi_detector = ImageROIDetector(token_path='keys/gdino_tocken.txt')
    # roi_detector.gen_rois(img_path, out_data_dir)
