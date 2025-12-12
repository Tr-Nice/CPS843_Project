#This code has been adapted from https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Define the filepath to the images
images_path = 'images'

# Run batched inference on a list of images
results = model(source=images_path, stream=True, save=True, project='yolo11n')  # return a generator of Results objects

# Process results generator
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen

model = YOLO("yolov3u.pt")

# Run batched inference on a list of images
results = model(source=images_path, stream=True, save=True, project='yolov3u')  # return a generator of Results objects

# Process results generator
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen

model = YOLO("yolov8n.pt")

# Run batched inference on a list of images
results = model(source=images_path, stream=True, save=True, project='yolov8n')  # return a generator of Results objects

# Process results generator
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen