### NOTE: Sample usage :-
# python yolowebcam.py --yolo yolo-coco

# Import the necessary libraries
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, help="Path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.6, help="Minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="Threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Load the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Initialize the video stream
vs = cv2.VideoCapture(0)
print('Detecting camera...')
time.sleep(20)


def detect(inputFrame):
	(W, H) = (None, None)
	if W is None or H is None:
		(H, W) = inputFrame.shape[:2]

	blob = cv2.dnn.blobFromImage(inputFrame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# Initialize lists of detected bounding boxes, confidences, and class IDs
	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > args["confidence"]:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(inputFrame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(inputFrame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return inputFrame


while(vs.isOpened()):
	(re, frame) = vs.read()

	if not re:
		break

	canvas = detect(frame)
	cv2.imshow('Video', canvas)

	if cv2.waitKey(30) & 0xFF == ord('q'):
		break

vs.release()
cv2.destroyAllWindows()
