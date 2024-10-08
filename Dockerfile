# Use the official Python 3.9 image from the Docker Hub
FROM python:3.9

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install PyTorch, torchvision, and torchaudio with CUDA 12.4 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4.1/index.html

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Set the working directory
WORKDIR /app

# Install the dependencies from requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --no-cache-dir mmsegmentation@git+https://github.com/open-mmlab/mmsegmentation.git@186572a3ce64ac9b6b37e66d58c76515000c3280
RUN pip install huggingface-hub

ARG GROQ_API_KEY
ENV GROQ_API_KEY=$GROQ_API_KEY

# Copy the application code into the container
COPY . .

RUN mkdir -p /input /output

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]