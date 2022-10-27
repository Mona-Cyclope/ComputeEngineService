from ComputeEngineManagerMTCMPS import ComputeEngine
import os
import numpy as np
from Yolov3.utils import load_yolo, detect_yolo, get_last_file

class ComputeEngineYoloV3(ComputeEngine):
    
    def __init__(self, cfg_path, weights_path, meta_path, gpu_id=0, score_min=0.45):
        load_dict = { "cfg": cfg_path, "weights": weights_path, "meta": meta_path }
        self.load_dict = load_dict
        self.gpu_id = gpu_id
        self.score_min = score_min
    
    def load(self):
        res_dict = load_yolo(self.load_dict, gpu_id=self.gpu_id)
        self.net = res_dict['net']
        self.meta = res_dict['meta']
        
    def process(self, data):
        y = detect_yolo(self.net, self.meta, data, score_min=self.score_min)
        return y
    
if __name__ == "__main__":
    
    model_home = "/home/mviti/gits/git_prod/plate-matching-ds/models"
    cfg =  get_last_file(os.path.join(model_home, "lp_detection"),"*.cfg")
    weights = get_last_file(os.path.join(model_home, "lp_detection"),"*.weights")
    meta = get_last_file(os.path.join(model_home, "lp_detection"),"*.data")
    
    model = ComputeEngineYoloV3(cfg_path=cfg, weights_path=weights, meta_path=meta)
    model.load()
    print("YOLO V3 CE LOAD ======================================================> OK")