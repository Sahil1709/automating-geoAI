import streamlit as st
from PIL import Image
import torch
import numpy as np
import rasterio
from utils.visuals import load_raster, enhance_raster_for_visualization
from utils.llm import get_completion, get_detectors
from mmcv import Config
from mmseg.models import build_segmentor
from mmseg.datasets.pipelines import Compose, LoadImageFromFile
from mmseg.apis import init_segmentor
from model_inference import inference_segmentor, process_test_pipeline
from huggingface_hub import hf_hub_download
import matplotlib
from torch import nn
import os
import matplotlib.pyplot as plt

# Function to process the image and convert it to .tif
def process_image(image, output_path):
    img = Image.open(image)
    img.save(output_path, format='TIFF')

# Function to call the Prithvi model for inference
def call_prithvi_model(tif_path, chosen_detector):
    config_path=f'configs/{chosen_detector}.py'
    ckpt=f'models/{chosen_detector}.pth'
    finetuned_model = init_segmentor(Config.fromfile(config_path), ckpt, device="cuda")
    custom_test_pipeline = process_test_pipeline(finetuned_model.cfg.data.test.pipeline)
    result = inference_segmentor(finetuned_model, tif_path, custom_test_pipeline=custom_test_pipeline)
    return result

def show_image(image):
    st.write(f"Image input shape is {image.shape}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)
    st.pyplot(fig)

def show_results(result, tif_path):
    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    input_data_inference = load_raster(tif_path)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    ax[0].imshow(result[0], norm=norm, cmap="jet")
    ax[1].imshow(enhance_raster_for_visualization(input_data_inference))
    ax[1].imshow(result[0], cmap="jet", alpha=0.3, norm=norm)
    for subplot in ax:
        subplot.axis('off')

    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Image Inference with Prithvi Model")
    st.sidebar.title("Available Detectors")
    st.sidebar.write(get_detectors())
    
    user_input = st.text_input("Enter your query")
    uploaded_file = st.file_uploader("Upload an image...", type=["tif"])
    submit = st.button("Detect")

    if submit and not uploaded_file:
        st.error("Please upload an image")

    if submit and not user_input:
        st.error("Please enter a query")

    if submit and user_input and uploaded_file:

        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        tif_path = uploaded_file.name

        st.write(f"Model Output")
        st.write(get_completion(user_input))
        chosen_detector = get_completion(user_input)['detector']
        if chosen_detector is None:
            st.error("Sorry, currently we do not have the detector for your query")
            os.remove(tif_path)

            return

        # Display the uploaded image
        input_data_inference = load_raster(tif_path)
        raster_for_visualization = enhance_raster_for_visualization(input_data_inference)
        show_image(raster_for_visualization)

        # Convert the image to .tif
        # tif_path = 'converted_image.tif'
        # process_image(uploaded_file, tif_path)
        
        # Call the Prithvi model for inference
        with st.spinner('Model is Running...'):
            inference_result = call_prithvi_model(tif_path, chosen_detector)  

        # Display the result
        show_results(inference_result, tif_path)

        # Delete the uploaded file after processing
        os.remove(tif_path)

if __name__ == "__main__":
    main()