# GDP-3-Arribada

## Installation

- Clone the repository:
```
git clone git@github.com:SamDower/GDP-3-Arribada.git
cd GDP-3-Arribada
```
- Optional (but recommended): create a new virtual environment.
```
python -m venv venv
source ./venv/bin/activate
```
- Install the requirements:
```
pip install -r requirements.txt
```
- Download [MegaDetector v5 model](https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt). For example:
```
mkdir -p models
wget -O models/md_v5a.0.0.pt https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt
```

## Running the detector on a stream of images

Install FFmpeg first.

- Run `run_detector_stream.py` (unfinished).
```
python run_detector_stream.py \
    "models/md_v5a.0.0.pt" \
    0 \
    udp://localhost:12345 \
    --threshold 0.1 \
    --fps 1.0
```
  - `0` is the input stream link (0 for the first webcam)
  - `udp://localhost:12345` is the target output UDP address to stream to (maybe should let OBS read this, and then restream)
    - In the future should be able get it to work with say RTMP
    - Or maybe write to a local static file (picture or json)
  - `--threshold` is the detection threshold for the MegaDetector model
  - `--fps` is the target (maximum) FPS of the output stream. It should be greater (but not much greater, otherwise expect a high latency) than the average speed of inference.
  - (Optional, try if it makes inference speed faster) `--num_threads`, either -1 (default, one main thread), or greater than 0 (threads for inference)

If no error prints, then there should be a UDP stream set up at localhost:12345. You might need to wait a while for the buffer to fill up. To access it, I used
```
ffplay udp://localhost:12345
```

Currently this solution doesn't copy the audio (check if we need this) and doesn't use OBS, which shouldn't be hard to implement.

## Running the detector on a single image

- Set the python path.
```
export PYTHONPATH="$PYTHONPATH:/path/to/GDP-3-Arribada/cameratraps:/path/to/GDP-3-Arribada/ai4eutils:/path/to/GDP-3-Arribada/yolov5"
```
- Navigate to the `cameratraps` folder
```
cd cameratraps/
```
- Run the `detection/run_detector.py` script, passing it the model, image, and threshold. E.g:
```
python detection/run_detector.py "../models/md_v5a.0.0.pt" --image_file "images/fox.jpg" --threshold 0.1
```
