from ComputeEngineManagerMTCMPS import ComputeEngine, ComputeEngineManager
from abc import ABCMeta, abstractmethod
import numpy as np
import torch as th
import sys
import pathlib
import os
import numpy as np
import pprint as pp
import cv2
from YoloV5.models.common import DetectMultiBackend
from YoloV5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from YoloV5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from YoloV5.utils.plots import Annotator, colors, save_one_box
from YoloV5.utils.torch_utils import select_device, smart_inference_mode
    
class YoloV5ComputeEngine(ComputeEngine):
    
    def __init__(self, weights, device='cpu', half=True, dnn=False, imgsz=(640,640),
                 conf_thres=0.25, iou_thres=0.45, max_det=1000, agnostic_nms=False, classes=None):
        self.weights = weights
        self.device = device
        self.half = half
        self.dnn = dnn
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.classes = classes
    
    def load(self):
        weights = self.weights 
        device = self.device
        half = self.half
        dnn = self.dnn
        imgsz = self.imgsz
        
        device = select_device('cpu')
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.imgsz = imgsz
        model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
        self.model = model
        
        
    def preprocess(self, image_list: list) -> th.Tensor:
        # process batch
        image_list = [ cv2.resize(image, list(reversed(self.imgsz))) for image in image_list ]
        im = np.stack(image_list).transpose([0,3,1,2])
        
        # adapt input
        im = th.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im
    
    def process(self, im: th.Tensor) -> list:    
        # inference
        pred = self.model(im, augment=False, visualize=False)
        return pred  
        
    def postprocess(self, pred:list) -> list:
        # nms
        pred = non_max_suppression(pred, 
                                   conf_thres=self.conf_thres,
                                   iou_thres=self.iou_thres,
                                   classes=self.classes,
                                   agnostic=self.agnostic_nms,
                                   max_det=self.max_det)
        
        
        return pred
    
if __name__ == "__main__":
    
    image_0 = "/raid-dgx3/mviti/PatrolCare/train_data/patrolcare_v2/labels/PC_ROI_boxing_selected_16122021-batch-2-2021-12-22/reflets_2021_01_22T08_45_26_00_360.jpg"
    image_1 = "/raid-dgx3/mviti/PatrolCare/train_data/patrolcare_v2/labels/PC_ROI_boxing_selected_16122021-batch-2-2021-12-22/reflets_2021_01_22T08_46_15_00_904.jpg"
    
    # prepare batch
    image_0 = cv2.imread(image_0)
    image_1 = cv2.imread(image_1)
    image_list = [image_0, image_1]

    weights = "/home/mviti/gits/yolov5/runs/train/exp2/weights/best.pt"
    device= 'cpu'
    half = False
    dnn = False
    imgsz = (640,640)
    classes = None
    agnostic_nms = False
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    
    ce_yolov5 = YoloV5ComputeEngine(weights=weights, device=device, half=half, imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, agnostic_nms=agnostic_nms)
    ce_yolov5.load()
    im = ce_yolov5.preprocess(image_list)
    pred = ce_yolov5.process(im)
    pred = ce_yolov5.postprocess(pred)
    
    names = ce_yolov5.names
    annotator = Annotator(image_0, line_width=1, example=str(names))
    hide_labels = False
    hide_conf = False
    
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class
        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        annotator.box_label(xyxy, label, color=colors(c, True))

    print(pred)
    
    