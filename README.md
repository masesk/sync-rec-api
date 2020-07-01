# sync-rec-api

A Flask server that recieves images in base64 format, saves them, and pass them to a neural network for object reconition using OpenCV's YOLOv3.

# Setup

## Install OpenCV 
Follow the instructions on [OpenCV's tutorial](https://docs.opencv.org/master/da/df6/tutorial_py_table_of_contents_setup.html) to install and setup OpenCV on your OS.

## Install Dependencies

### Requirements:

1. Python3

### Instructions

1. (OPTIONAL) Generate a virtual enviroment to keep pip list install clean.
Run :
```
python3 -m venv .venv
```
2. (OPTIONAL) Activate virtual enviroment. Run:
```
. .venv/bin/activate
```
3. Install required packages. Run:
```
pip install -r requirements.txt
```

## Run Flask Server

1. To start the Flask server on port 8080, run:
```
python app.py
```
2. To change the port, modify port=8080 in app.run() function in app.py.
