# Custom Football Player Detection Dataset

## Football Dataset and Test Videos
[Football Player Detection Dataset](https://drive.google.com/drive/folders/1ltpD_EkmHnhU6i4KqR_Jypv9zpidQMLn?usp=sharing)

[Test Videos Match](https://drive.google.com/drive/folders/1SNUDDHVCw9xNunSelTQ0y1w93fFaEoHa?usp=sharing)

# Connect Google Colab with Google Drive to Read and Write

***Python Code***
### Connect with Google Drive
```
from google.colab import drive

drive.mount('/content/drive')
```

# YOLOv8: Video Object Detection with Python on Custom Dataset

***Python Code***

### Install YOLOv8

### Pip install (recommended)
!pip install ultralytics
 
import ultralytics

ultralytics.checks()    


### YOLO8 Training HyperParameters

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

# Training YOLOv8 for Player, Referee and Football Detection

### Training YOLO8 for Object Detection

In the provided code, the backslash ('\') is used to continue the command onto the next line for better readability. This is called line continuation.

In Python, a backslash at the end of a line indicates that the command continues on the next line. It's commonly used when a single line of code becomes too long and you want to split it into multiple lines for improved readability.

!yolo task=detect \

    mode=train \
    
    model=yolov8n.pt \
    
data=/content/drive/MyDrive/ObjectDetection/FootballPlayerDetection/dataset.yaml \

    epochs=10 \
    
    imgsz=1920 \
    
    batch=5 \
    
project=/content/drive/MyDrive/ObjectDetection/FootballPlayerDetection/TrainingResults \

name=footballDetection
