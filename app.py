import streamlit as st
from PIL import Image
import torch
import numpy as np
import rasterio
from utils.visuals import load_raster, enhance_raster_for_visualization, show_results, show_image
from utils.llm import get_completion, get_detectors
from utils.data_collection import get_sentinel_data
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
import matplotlib.patches as mpatches
import subprocess

# Define the classes and their corresponding colors
CLASSES = (
    "Natural Vegetation",
    "Forest",
    "Corn",
    "Soybeans",
    "Wetlands",
    "Developed/Barren",
    "Open Water",
    "Winter Wheat",
    "Alfalfa",
    "Fallow/Idle Cropland",
    "Cotton",
    "Sorghum",
    "Other",
)

CLASS_COLORS = {
    "Natural Vegetation": (0, 128, 0),  # Green
    "Forest": (34, 139, 34),            # Forest Green
    "Corn": (255, 255, 0),              # Yellow
    "Soybeans": (0, 255, 0),            # Lime
    "Wetlands": (0, 0, 255),            # Blue
    "Developed/Barren": (128, 128, 128),# Gray
    "Open Water": (0, 191, 255),        # Deep Sky Blue
    "Winter Wheat": (255, 165, 0),      # Orange
    "Alfalfa": (255, 20, 147),          # Deep Pink
    "Fallow/Idle Cropland": (139, 69, 19), # Saddle Brown
    "Cotton": (255, 255, 255),          # White
    "Sorghum": (255, 0, 0),             # Red
    "Other": (128, 0, 128),             # Purple
}

def display_class_colors():
    st.title("Class Colors")
    fig, ax = plt.subplots(figsize=(10, 5))
    patches = [mpatches.Patch(color=np.array(color)/255.0, label=cls) for cls, color in CLASS_COLORS.items()]
    ax.legend(handles=patches, loc='center', ncol=2)
    ax.axis('off')
    st.pyplot(fig)

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

# def show_results(result, tif_path):
#     fig, ax = plt.subplots(1, 2, figsize=(8, 8))
#     input_data_inference = load_raster(tif_path)
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
#     ax[0].imshow(result[0], norm=norm, cmap="jet")
#     ax[1].imshow(enhance_raster_for_visualization(input_data_inference))
#     ax[1].imshow(result[0], cmap="jet", alpha=0.3, norm=norm)
#     for subplot in ax:
#         subplot.axis('off')

#     st.pyplot(fig)

# Streamlit app
def main():
    st.title("Image Inference with Prithvi Model")
    st.sidebar.title("Available Detectors")
    st.sidebar.write(get_detectors())

    test = st.sidebar.button("test")
    if test:
        assets = get_sentinel_data(-72.5, 40.5, "2020-01-01", "2020-01-20")
        st.write(assets)   
        for item in assets:
            input_data_inference = load_raster(item.href)
            raster_for_visualization = enhance_raster_for_visualization(input_data_inference)
            show_image(raster_for_visualization)
            # with rasterio.open(item.href) as dataset:
            #     rasterio.plot.show(dataset)

    
    user_input = st.text_input("Enter your query")
    uploaded_file = st.file_uploader("Upload an image...", type=["tif"])
    submit = st.button("Detect")

    if submit and not uploaded_file:
        st.error("Please upload an image")

    if submit and not user_input:
        st.error("Please enter a query")

    if submit and user_input and uploaded_file:

        with open(f'input/{uploaded_file.name}', "wb") as f:
            f.write(uploaded_file.getbuffer())

        tif_path = 'input/' + uploaded_file.name

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
            # inference_result = call_prithvi_model(tif_path, chosen_detector)  
            config_path = f'configs/{chosen_detector}.py'
            ckpt_path = f'models/{chosen_detector}.pth'
            output_path = 'output/'
            command = [
                'python', 'model_inference.py',
                '-config', config_path,
                '-ckpt', ckpt_path,
                '-input', 'input/',
                '-output', output_path,
                '-input_type', 'tif',
                '-bands', '0', '1', '2', '3', '4', '5'
            ]
            subprocess.run(command, check=True)

        # Display the result
        result_path = os.path.join(output_path, uploaded_file.name.replace('.tif', '_pred.tif'))
        inference_result = load_raster(result_path)

        show_results(tif_path, result_path)

        if chosen_detector == 'multi_temporal_crop_classification_Prithvi_100M':
            display_class_colors()

        # Delete the uploaded file after processing
        os.remove(tif_path)

if __name__ == "__main__":
    main()