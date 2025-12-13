#This code has been adapted from https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
from ultralytics import YOLO
# Define the filepath to the images
images_path = 'images'

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(source=images_path, stream=False, save=True, project='yolo11n')  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen