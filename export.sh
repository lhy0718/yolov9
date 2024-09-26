yolo export model=yolov9t.pt format=edgetpu imgsz=320
# WARNING ⚠️ argument '--help' does not require leading dashes '--', updating to 'help'.
# Arguments received: ['yolo', 'export', '--help']. Ultralytics 'yolo' commands use the following syntax:
#     yolo TASK MODE ARGS

#     Where   TASK (optional) is one of {'segment', 'obb', 'detect', 'pose', 'classify'}
#             MODE (required) is one of {'predict', 'track', 'benchmark', 'train', 'val', 'export'}
#             ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
#                 See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

# 1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
#     yolo train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01

# 2. Predict a YouTube video using a pretrained segmentation model at image size 320:
#     yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

# 3. Val a pretrained detection model at batch-size 1 and image size 640:
#     yolo val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640

# 4. Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
#     yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

# 5. Explore your datasets using semantic search and SQL with a simple GUI powered by Ultralytics Explorer API
#     yolo explorer data=data.yaml model=yolov8n.pt

# 6. Streamlit real-time webcam inference GUI
#     yolo streamlit-predict
    
# 7. Run special commands:
#     yolo help
#     yolo checks
#     yolo version
#     yolo settings
#     yolo copy-cfg
#     yolo cfg

# Docs: https://docs.ultralytics.com
# Community: https://community.ultralytics.com
# GitHub: https://github.com/ultralytics/ultralytics