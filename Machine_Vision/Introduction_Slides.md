## ***The Materials in this lesson are from: YOLOv8: Video Object Detection with Python on Custom Dataset***
***Created by: Dr. Mazhar Hussain | Deep Learning, Computer Vision, AI & Python | CS Lecturer***

# Introduction to Object Detection

![image](https://github.com/user-attachments/assets/46aff9b5-6b45-4a1d-85cd-b8eb1cb9fbc5)

![image](https://github.com/user-attachments/assets/89e803f2-7124-4cfe-9967-ef8c142f971c)

![image](https://github.com/user-attachments/assets/bc67b6c3-5a93-43ae-9a6a-d94a583a768a)

# Introduction to YOLO

![image](https://github.com/user-attachments/assets/03985ee9-7e9d-4fee-b0d2-dc123f61fc1d)

![image](https://github.com/user-attachments/assets/36f25c57-c04a-4d7f-97be-0231a3426e79)

![image](https://github.com/user-attachments/assets/0c274fee-a365-4998-ba18-02dc87ef1a07)

![image](https://github.com/user-attachments/assets/dd7d3b7d-c920-4c25-9d93-69577dcbd3fd)

# How YOLO works?

![image](https://github.com/user-attachments/assets/8ce9bbfd-5e6e-4e62-99b3-df51a38b9697)

![image](https://github.com/user-attachments/assets/dbddff0d-de32-4d19-b54b-9c1776be6a89)

![image](https://github.com/user-attachments/assets/b774ea96-8f2d-4468-a927-44582e08a564)

![image](https://github.com/user-attachments/assets/70ebd056-58e1-4808-9b37-279db47acf07)

![image](https://github.com/user-attachments/assets/f585b6dc-647e-4488-a9c2-f2b54d6891db)

![image](https://github.com/user-attachments/assets/1d03348d-e05c-49a8-81df-28b379d52b91)

![image](https://github.com/user-attachments/assets/0ebb01c3-c4e7-42f1-aee5-0c51c619fcfb)

![image](https://github.com/user-attachments/assets/29056697-5c7f-4593-955b-e053c9f400a2)

![image](https://github.com/user-attachments/assets/e7d57884-15dc-4ce1-85d8-413d3da4da00)

# YOLO v8

![image](https://github.com/user-attachments/assets/605cc9ff-9b44-4ae6-a279-4d8cea478e72)

![image](https://github.com/user-attachments/assets/75158312-b927-4742-b212-a0a1f63031e8)

![image](https://github.com/user-attachments/assets/3053a975-e32c-4c1c-a9a2-29563a05e20d)

![image](https://github.com/user-attachments/assets/3f17a4f6-e2b3-4b31-b297-80973d0c69a1)

![image](https://github.com/user-attachments/assets/64e5f65a-289e-413f-8d07-6e52c9c236be)

# YOLOv8 Object Detection with Python

YOLOv8 Object Detection with Python
YOLOv8 models are fast, accurate, and easy to use, making them ideal for real-time object detection task trained on large datasets and run on diverse hardware platforms, from CPUs to GPUs. Object detection involves identifying the location and class of objects in an image or video stream. The output is a set of bounding boxes that enclose the objects in the image, along with class labels and confidence scores for each box. Object detection is a perfect choice when you need to detect and identify objects of interest, but don’t need to know exactly where the object is or its exact shape.

YOLOv8 detection models have no suffix and are the default YOLOv8 models, i.e. (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt) and are pretrained on COCO dataset with the following Classes.

['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

YOLOv8 pretrained Detect models (nano, small, medium, large and extra large based on number of parameters) are shown in the table below:

![image](https://github.com/user-attachments/assets/80f4ab42-21f1-4793-9243-d43a48c87e2a)

## Setup UltraLytics for YOLOv8

%pip install ultralytics
import ultralytics
ultralytics.checks()

Load YOLOv8 for Object Detection
from ultralytics import YOLO
 
## Load a model
**# You can use different YOLOv8 variants (yolov8n, yolov8s, yolov8m, yolov8l, yolov8nx)**
model = YOLO('yolov8n.pt')  # load a pretrained model
 
# Use the model
results = model('https://ultralytics.com/images/zidane.jpg')  # predict on an image
# Save the output image after detection 
results[0].save('/content/output.jpg')
# Print the COCO dataset classes on which model is trained.
print(model.names.values())
Input Image
