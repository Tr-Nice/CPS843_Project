#This code has been adapted from https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
from ultralytics import YOLO
# Define the filepath to the images
images_path = 'images'

model = YOLO("yolov3u.pt") # pretrained YOLO3u model

for i in range(0,5):
    # Run batched inference on a list of images
    results = model(source=images_path, stream=False, save=True, save_txt=True, save_conf=True, project='yolov3u')  # return a generator of Results objects

    # Process results generator
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen    