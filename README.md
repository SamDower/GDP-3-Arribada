# README - GDP 3 Arribada

## Installation

- Navigate to the directory you want to install in:
```
cd /directory/to/install/in/
```
- Clone the repository:
```
git clone --recurse-submodules git@github.com:SamDower/GDP-3-Arribada.git
cd GDP-3-Arribada
```
- Create a new virtual environment: (Optional but recommended)
```
python -m venv venv
source ./venv/bin/activate
```
- Install the requirements:
  - Note that installing packages such as Torch may take a while, run with the `--verbose` flag if you get nervous when a computer looks like it's doing nothing for too long.
```
pip install -r requirements.txt
```
- Download detector model:
  - We recommend using the Microsoft [MegaDetector v5 model](https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt). 
  - Other models can be used instead, see the "Running the detector section" for how to use a different model. 
```
mkdir -p models
wget -O models/md_v5a.0.0.pt https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt
```

- Install FFmpeg:
```
sudo apt install ffmpeg
```
---
---

## Running an example stream

Stream an example video to your local machine for demonstration.
```
ffmpeg -re -r 30 -i sample.mp4 -f mpegts udp://localhost:9999
```
- `-re` flag forces video to be streamed in real time.
- `-r 30` flag sets the stream framerate to 30fps.
- `-i sample.mp4` flag sets the input file as sample.mp4
  - Change the file path to any video file in the same directory to demo another stream.
- `-f mpegts` flag sets the output format to MPEG Transport Stream.
- `udp://localhost:9999` is the address to stream to.
  - This sends the stream to the local machine on port 9999.
  - If a "failed to bind" error occurs, change the port to one not currently in use.

---

## Running the detector

Run the detector model on a video stream, by default the local demonstration stream.
```
python run_detector_stream.py \
    "models/md_v5a.0.0.pt" \
    udp://localhost:9999 \
    udp://localhost:9995 \
    --threshold 0.1 \
    --fps 10.0
```
  - `"models/md_v5a.0.0.pt"` path to the detection model parameters.
    - If a different model is used, change this argument to point to it.
  - `udp://localhost:9999` link to the input stream.
    - This is the example stream link from the previous part and needs to match the address on the running stream.
    - Change this address to point to the stream you want to run the detector on.
    - This can also be set to point to a static video file. 
  - `udp://localhost:9995` target to stream ouput to.
    - This will stream to the local machine on port 9995.
    - If a "failed to bind" error occurs, change the port to one not currently in use.
    - Change this address to the intended streaming destination.
    - This can also be set to a file path, in which case it will write the processed stream to the specified file. 
  - `--threshold` is the detection threshold for the MegaDetector model.
    - 0.0 will accept any box the model predicts.
    - 1.0 will only accept boxes the model is 100% confident with.
    - We reccomend 0.1 as a reasonable parameter.
  - `--fps` is the maximum framerate of the output stream. 
    - This should be greater than the average speed of inference (10.0 on average modern GPUs).
    - Setting this too high may lead to increased restreaming latency.
  - `--num_threads` is the number of threads dispatched to process detection.
    - This is highly system specific, and requesting any number other than the default (1) may lead to crashes.

---

## Viewing the output stream

If you have set up to restream to the local machine, view the output stream by running the following:
```
ffplay udp://localhost:9995
```
- `udp://localhost:9995` link to the output stream.
    - This is the restreamed link from the previous part and needs to match the address on the running detection stream.
    - If you want to view the unprocessed stream, change the parameter to the input stream address.
    - Note that if you try to view the input stream while using it as input for the detector you will get a "failed to bind" error. Disable the detector then try again.

---
---

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
