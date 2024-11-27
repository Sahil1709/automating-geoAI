import streamlit as st
from utils.visuals import load_raster, enhance_raster_for_visualization, show_image

st.title("Raster Visualizer")
uploaded_file = st.file_uploader("Upload a .tif file", type=["tif"])
if uploaded_file is not None:
    with open(f'input/{uploaded_file.name}', "wb") as f:
        f.write(uploaded_file.getbuffer())

    tif_path = 'input/' + uploaded_file.name

    # Display the uploaded image
    input_data_inference = load_raster(tif_path)
    raster_for_visualization = enhance_raster_for_visualization(input_data_inference)
    show_image(raster_for_visualization)