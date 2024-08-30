docker run --name yolov9 -it\
    -v /home/hanyong/yolov9:/workspace\
    --gpus all\
    --ipc=host\
    --ulimit memlock=-1\
    --ulimit stack=67108864\
    nvcr.io/nvidia/pytorch:21.11-py3