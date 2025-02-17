import os
import time
import argparse

from tqdm.auto import tqdm
from PIL import Image, ImageDraw
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

import streamlit as st

from image_retrival import ImageRetrival


def get_args():
    parser = argparse.ArgumentParser(description='Merge hdf5 files.')
    parser.add_argument('--dataset_name',
                        type=str,
                        help='Name of created dataset.')
    parser.add_argument('--gdino_tocken_path',
                        type=str,
                        default='keys/gdino_tocken.txt',
                        help='Name of created dataset.')
    
    args = parser.parse_args()
    return args


args = get_args()
dataset_name = args.dataset_name
retrival = ImageRetrival(gdino_tocken_path=args.gdino_tocken_path)


def main():
    global retrival, dataset_name
    st.title("Image Similarity Search (PyTorch)")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:

        query_image = Image.open(uploaded_file)
        st.image(query_image, caption="Uploaded Image", use_container_width=True)
        tmp_img_path = f'./data/q.jpg'
        query_image.save(tmp_img_path)

    query_prompt = st.text_input("Enter prompt[optional]", value='')
    imgs_per_crop = st.number_input("Num of similar images per crop", min_value=0, max_value=100, value=5)
    
    detection_mode = st.radio("Search mode:", ('superglobal', 'ames'), index=0)
    
    if st.button("Start Inference"):
        if uploaded_file is None:
            st.warning("Please upload an image first.") 
            print('you need to upload image')
            return

        total_time = time.time()
        imgs_dict = retrival.infer(tmp_img_path, query_prompt, dataset_name=dataset_name, imgs_per_crop=imgs_per_crop, detection_mode=detection_mode)
        total_time = time.time() - total_time
        st.text(f"Execution time: {total_time:.2f} s")
        
        st.title("Similar Image Finder")

        for query_image_path, similar_image_paths in \
                sorted(imgs_dict.items(), key=lambda item: item[0], reverse=False):
            col1, col2 = st.columns([2, 8])

            with col1:
                query_image = Image.open(query_image_path)
                st.image(query_image, caption="Query Image", use_container_width=True)

            with col2:
                st.subheader("Similar Images:")
                num_cols = 2
                with st.container():
                    cols = st.columns(num_cols)
                    for i, similar_image_path in enumerate(similar_image_paths):
                        with cols[i % num_cols]:
                            similar_image = Image.open(similar_image_path)
                            name = similar_image_path.split('/')[-1]
                            st.image(similar_image, caption=name, use_container_width=True)
            st.markdown("---")


if __name__ == "__main__":
    main()