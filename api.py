from fastapi import FastAPI, File, UploadFile, Form
from typing import Annotated
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, time, uvicorn, base64
import subprocess
from utils.visuals import load_raster, enhance_raster_for_visualization, show_results
from utils.llm import get_completion, get_detectors
from model_inference import inference_segmentor, process_test_pipeline
from mmcv import Config
from mmseg.apis import init_segmentor
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI()

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/detectors/")
async def return_detectors():
    return {"detectors": get_detectors()}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return JSONResponse(content={"filename": file.filename})

@app.get("/upload/")
async def upload_page():
    return HTMLResponse(open("static/index.html").read())

@app.post("/process/")
async def process_file(file: UploadFile, detector: str = Form(...)):

    # Save the uploaded file
    input_dir = 'input/'
    output_dir = 'output/'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(input_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Check if the detector is valid
    if detector not in get_detectors():
        return {"error": "Invalid detector name"}

    # Run the model inference
    config_path = f'configs/{detector}.py'
    ckpt_path = f'models/{detector}.pth'
    command = [
        'python', 'model_inference.py',
        '-config', config_path,
        '-ckpt', ckpt_path,
        '-input', input_dir,
        '-output', output_dir,
        '-input_type', 'tif',
        '-bands', '0', '1', '2', '3', '4', '5'
    ]
    
    start_time = time.time()
    subprocess.run(command, check=True)
    end_time = time.time()
    inference_time = end_time - start_time

    # Get the result file path
    result_path = os.path.join(output_dir, file.filename.replace('.tif', '_pred.tif'))

    # Return the result
    # return {
    #     "input_file": file.filename,
    #     "output_file": os.path.basename(result_path),
    #     "inference_time": inference_time,
    #     "file_response": FileResponse(result_path, media_type='image/tiff', filename=os.path.basename(result_path))
    # }

    return FileResponse(result_path, media_type='image/tiff', filename=os.path.basename(result_path))


@app.post("/process-base64/")
async def process_file_base64(file: str = Form(...), detector: str = Form(...)):

    # Decode the base64 encoded file
    file_data = base64.b64decode(file)
    input_dir = 'input/'
    output_dir = 'output/'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(input_dir, "uploaded_file.tif")
    with open(file_path, "wb") as f:
        f.write(file_data)

    # Check if the detector is valid
    if detector not in get_detectors():
        return {"error": "Invalid detector name"}

    # Run the model inference
    config_path = f'configs/{detector}.py'
    ckpt_path = f'models/{detector}.pth'
    command = [
        'python', 'model_inference.py',
        '-config', config_path,
        '-ckpt', ckpt_path,
        '-input', input_dir,
        '-output', output_dir,
        '-input_type', 'tif',
        '-bands', '0', '1', '2', '3', '4', '5'
    ]
    
    start_time = time.time()
    subprocess.run(command, check=True)
    end_time = time.time()
    inference_time = end_time - start_time

    # Get the result file path
    result_path = os.path.join(output_dir, "uploaded_file_pred.tif")

    # Return the result
    return FileResponse(result_path, media_type='image/tiff', filename=os.path.basename(result_path))

def delete_directories():
    input_dir = 'input/'
    output_dir = 'output/'
    print(f"\033[92m{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\033[0m: Deleting files in input and output directories")
    if os.path.exists(input_dir):
        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

# Initialize the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(delete_directories, 'interval', minutes=5)
scheduler.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)