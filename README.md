# GDP-3-Arribada

## Installation

- Clone the repository:
```
git clone git@github.com:SamDower/GDP-3-Arribada.git
```
- Optional (but recommended): create a new virtual environment.
- Install the requirements:
```
pip install -r requirements.txt
```

## Running the detector on a single image

- Set the python path.
```
export PYTHONPATH="$PYTHONPATH:cameratraps:ai4eutils:yolov5"
```
- Navigate to the `cameratraps` folder/
```
cd cameratraps/
```
- Run the `detection/run_detector.py` script, passing it the model, image, and threshold. E.g:
```
python detection/run_detector.py "$HOME/megadetector/md_v5a.0.0.pt" --image_file "images/fox.jpg" --threshold 0.1
```