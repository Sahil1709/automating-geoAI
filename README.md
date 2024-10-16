# Automating GeoAI Workflow


This project is designed to take user queries, utilize a large language model (LLM) agent to map those queries to the available foundation models, and display the results back to the user. By leveraging advanced AI capabilities, the system ensures accurate and efficient query processing, providing users with relevant and insightful results based on the detectors at hand.

# Setup and Run Instructions

1. **Download Models & Update docker file**
    - Download models from [Hugging Face IBM NASA Geospatial](https://huggingface.co/ibm-nasa-geospatial).
    - Save the models to the `models` directory.
    - Save the configurations to the `configs` directory.
    - Then Add these 2 lines at the start after line 30 in Dockerfile
    ```
    COPY /path/to/config /app/config
    COPY /path/to/checkpoint /app/checkpoint
    ```

2. **Build Docker Image**
    ```sh
    docker build -t geoai-app --build-arg GROQ_API_KEY=[value] .
    ```

    ### The time it takes
    ```
    [+] Building 1409.0s (12/12) FINISHED                                                                    docker:default
    => [internal] load build definition from Dockerfile                                                               0.0s
    => => transferring dockerfile: 972B                                                                               0.0s
    => [internal] load metadata for docker.io/library/python:3.9                                                      0.8s
    => [internal] load .dockerignore                                                                                  0.0s
    => => transferring context: 2B                                                                                    0.0s
    => CACHED [1/7] FROM docker.io/library/python:3.9@sha256:a23efa04a7f7a881151fe5d473770588ef639c08fd5f0dcc6987dbe  0.0s
    => [internal] load build context                                                                                  0.1s
    => => transferring context: 7.57kB                                                                                0.0s
    => [2/7] RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118        386.2s
    => [3/7] RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4.1/index.html      916.1s
    => [4/7] COPY requirements.txt /app/requirements.txt                                                              0.8s
    => [5/7] WORKDIR /app                                                                                             0.2s
    => [6/7] RUN pip install -r requirements.txt                                                                     58.7s
    => [7/7] COPY . .                                                                                                 0.5s
    => exporting to image                                                                                            45.3s
    => => exporting layers                                                                                           45.3s
    => => writing image sha256:618f583c54235566c81d634150de086bca8787377ac730bbace62bf0c9e62fe5                       0.0s
    => => naming to docker.io/library/geoai-app                                                                       0.0s
    ```


3. **Run the Application**
    ```sh
    docker run -p 8501:8501 geoai-app
    ```
