# torchscript export
# python export.py\
#     --weights yolov9-c-converted.pt\
#     --img-size 640\
#     --device cpu\
#     --inplace\
#     --optimize

# tf.js export
python export.py\
    --weights yolov9-c-converted.pt\
    --img-size 640\ 
    --device 0\
    --half\
    --int8\
    --topk-all 20\
    --include tfjs
