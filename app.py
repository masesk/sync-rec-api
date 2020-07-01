from flask import Flask
from flask import request
from flask import jsonify
import cv2 as cv

import numpy as np
import base64
import io
import json
from imageio import imread
from PIL import Image

app = Flask(__name__)

confThreshold = 0.5  
nmsThreshold = 0.4   
inpWidth = 256      
inpHeight = 256      


        

classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):

    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom, frame):
    
    cv.rectangle(frame, (left, top), (right, bottom), (20, 179, 30), 1)
    
    label = '%.2f' % conf
        
    
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]


    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])


    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)

@app.route("/",  methods = ["POST"])
def process_base64():
    b64_string=request.json["data"]
    imgdata = base64.b64decode(b64_string)
    filename = 'raw.jpg' 
    with open(filename, 'wb') as f:
        f.write(imgdata)

    basewidth = 500
    img = Image.open('raw.jpg')
    
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize))
    img=img.rotate(270, expand=True)
    img.save('resized.jpg') 
    frame = cv.imread('resized.jpg')
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Write the frame with the detection boxes

    retval, buffer = cv.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    return jsonify({"data": jpg_as_text.decode("utf-8")})
    
@app.route("/checkstatus")
def check():
    return ({"up": 200})

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080)