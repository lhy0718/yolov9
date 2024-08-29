python export.py --weights yolov9-t-converted.pt --imgsz 640 --batch-size 1 --include edgetpu

# usage: export.py [-h] [--data DATA] [--weights WEIGHTS [WEIGHTS ...]] [--imgsz IMGSZ [IMGSZ ...]] [--batch-size BATCH_SIZE] [--device DEVICE] [--half] [--inplace] [--keras] [--optimize] [--int8]
#                  [--dynamic] [--simplify] [--opset OPSET] [--verbose] [--workspace WORKSPACE] [--nms] [--agnostic-nms] [--topk-per-class TOPK_PER_CLASS] [--topk-all TOPK_ALL]
#                  [--iou-thres IOU_THRES] [--conf-thres CONF_THRES] [--include INCLUDE [INCLUDE ...]]

# optional arguments:
#   -h, --help            show this help message and exit
#   --data DATA           dataset.yaml path
#   --weights WEIGHTS [WEIGHTS ...]
#                         model.pt path(s)
#   --imgsz IMGSZ [IMGSZ ...], --img IMGSZ [IMGSZ ...], --img-size IMGSZ [IMGSZ ...]
#                         image (h, w)
#   --batch-size BATCH_SIZE
#                         batch size
#   --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
#   --half                FP16 half-precision export
#   --inplace             set YOLO Detect() inplace=True
#   --keras               TF: use Keras
#   --optimize            TorchScript: optimize for mobile
#   --int8                CoreML/TF INT8 quantization
#   --dynamic             ONNX/TF/TensorRT: dynamic axes
#   --simplify            ONNX: simplify model
#   --opset OPSET         ONNX: opset version
#   --verbose             TensorRT: verbose log
#   --workspace WORKSPACE
#                         TensorRT: workspace size (GB)
#   --nms                 TF: add NMS to model
#   --agnostic-nms        TF: add agnostic NMS to model
#   --topk-per-class TOPK_PER_CLASS
#                         TF.js NMS: topk per class to keep
#   --topk-all TOPK_ALL   ONNX END2END/TF.js NMS: topk for all classes to keep
#   --iou-thres IOU_THRES
#                         ONNX END2END/TF.js NMS: IoU threshold
#   --conf-thres CONF_THRES
#                         ONNX END2END/TF.js NMS: confidence threshold
#   --include INCLUDE [INCLUDE ...]
#                         torchscript, onnx, onnx_end2end, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle