import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
outputLayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# loading images
imagName = input("Enter file name:")
img = cv2.imread(imagName)
img = cv2.resize(img, None, fx= 0.7,fy=0.6)
#cv2.namedWindow("Image", cv2.WINDOW_FULLSCREEN)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)


net.setInput(blob)
outs = net.forward(outputLayers)

# showing information on the screen
boxes = []
confidences = []
classIds = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence > 0.1:
            # object detected
            centerX = int(detection[0] * width)
            centerY = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(centerX - w/2)
            y = int(centerY - h/2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            classIds.append(classId)


numObjectsDetected = len(boxes)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
count = 0
for i in range(numObjectsDetected):
    if i in indexes:
        x, y, w, h = boxes[i]
        tmp = str(round(confidences[i], 3))
        label = str(classes[classIds[i]])
        Str = label + tmp
        count += 1

        if count < 55:
            color = colors[i]
        else:
            color = (0, 0, 0)


        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, Str, (x, y + 30), font, 1, color, 1)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
