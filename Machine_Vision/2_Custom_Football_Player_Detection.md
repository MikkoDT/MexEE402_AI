# Custom Football Player Detection Dataset

## Football Dataset and Test Videos
[Football Player Detection Dataset](https://drive.google.com/drive/folders/1ltpD_EkmHnhU6i4KqR_Jypv9zpidQMLn?usp=sharing)

[Test Videos Match](https://drive.google.com/drive/folders/1SNUDDHVCw9xNunSelTQ0y1w93fFaEoHa?usp=sharing)

# Connect Google Colab with Google Drive to Read and Write

***Python Code***
### Connect with Google Drive
from google.colab import drive

drive.mount('/content/drive')

# YOLOv8: Video Object Detection with Python on Custom Dataset

***Python Code***

### Install YOLOv8

### Pip install (recommended)
!pip install ultralytics
 
import ultralytics

ultralytics.checks()    


# YOLO8 Training HyperParameters

"""
 
task= detect or segment
 
mode= train, val
 
model= path to model file, i.e. yolov8n.pt, yolov8n.yaml
 
data= path to data file dataset.yaml
 
epochs= number of epochs to train for
 
imgsz= size of input images as integer
 
batch= number of images per batch
 
project= project traing results saving path
 
name= experiment name
"""
