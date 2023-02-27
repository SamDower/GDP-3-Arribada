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
