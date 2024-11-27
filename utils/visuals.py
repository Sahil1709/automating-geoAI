import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
import streamlit as st
import yaml

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

def load_raster(path, crop=None):
    with rasterio.open(path) as src:
        img = src.read()

        # load first 6 bands
        img = img[:6]

        img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
        if crop:
            img = img[:, -crop[0]:, -crop[1]:]
    return img

def enhance_raster_for_visualization(raster, ref_img=None):
    if ref_img is None:
        ref_img = raster
    channels = []
    for channel in range(raster.shape[0]):
        valid_mask = np.ones_like(ref_img[channel], dtype=bool)
        valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
        mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
        normalized_raster = (raster[channel] - mins) / (maxs - mins)
        normalized_raster[~valid_mask] = 0
        clipped = np.clip(normalized_raster, 0, 1)
        channels.append(clipped)
    clipped = np.stack(channels)
    channels_last = np.moveaxis(clipped, 0, -1)[..., :3]
    rgb = channels_last[..., ::-1]
    return rgb

def plot_image_mask_reconstruction(normalized, mask_img, pred_img):
    # Mix visible and predicted patches
    rec_img = normalized.clone()
    rec_img[mask_img == 1] = pred_img[mask_img == 1]  # binary mask: 0 is keep, 1 is remove

    mask_img_np = mask_img.numpy().reshape(6, 224, 224).transpose((1, 2, 0))[..., :3]

    rec_img_np = (rec_img.numpy().reshape(6, 224, 224) * stds) + means
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))

    for subplot in ax:
        subplot.axis('off')

    ax[0].imshow(enhance_raster_for_visualization(input_data))
    masked_img_np = enhance_raster_for_visualization(input_data).copy()
    masked_img_np[mask_img_np[..., 0] == 1] = 0
    ax[1].imshow(masked_img_np)
    ax[2].imshow(enhance_raster_for_visualization(rec_img_np, ref_img=input_data))


def show_results(input_raster_path, output_raster_path):
    input_data = load_raster(input_raster_path)
    input_raster_for_visualization = enhance_raster_for_visualization(input_data)

    # Load the second raster (output/prediction)
    output_data = load_raster(output_raster_path)
    output_raster_for_visualization = enhance_raster_for_visualization(output_data)

    # Define a colormap and normalization for classes (e.g., 0, 1, 2 for 3 classes)
    norm = mcolors.Normalize(vmin=0, vmax=2)  # Assuming 3 classes (0, 1, 2)
    cmap = plt.cm.jet  # You can change this to any colormap

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Single plot for overlay

    # Turn off axis
    ax.axis('off')

    # Display the input raster
    ax.imshow(input_raster_for_visualization)

    # Overlay the output raster with some transparency
    im = ax.imshow(output_raster_for_visualization, cmap=cmap, norm=norm, alpha=0.3)

    # Add a colorbar to represent the classes (legend)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    # Set the class labels for the colorbar
    cbar.set_ticks([0, 1, 2])  # Assuming you have 3 classes: 0, 1, 2
    cbar.set_ticklabels(['Class 0', 'Class 1', 'Class 2'])  # Change labels as needed

    # Show the final result
    plt.show()

    st.pyplot(fig)

def show_image(image):
    st.write(f"Image input shape is {image.shape}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)
    st.pyplot(fig)