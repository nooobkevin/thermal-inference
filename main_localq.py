"""
对之前乱七八糟的代码进行模块化重构
工具函数模块化后放进utils
"""
import sys
import socket
import subprocess as sp
import threading
import numpy as np
import cv2
import struct
import time
from senxor.utils import (
    data_to_frame, cv_render, cv_filter
)
sys.path.append("Models")
from models.YOLOnnx import YOLOModel
from utils.preprocess import scale_coords

import matplotlib.image

from senxor.mi48 import MI48, format_header, format_framestats
from senxor.utils import (
    data_to_frame, remap, cv_filter, cv_render, RollingAverageFilter, connect_senxor
)
from utils.datamaker import DataMaker
from utils.detection import HumanDetector
from faker import Faker
"""
把thermal推流服务封装成类
"""

name_dic = {'-1': 'Identifying...'}
fake = Faker()
class ThermalServer:
    def __init__(self, push_stream=None, server=None, model=None) -> None:
        self.push_stream = push_stream  # Push stream server
        self.server = server  # Update the inference results to this server
        
        self.model = model  # The model used to inference

        self.nowpic = None
        self.pic_lock = threading.Lock()
        self.color = [(0,255,0), (0,0,255), (255,0,0)]
        self.status = "normal"
        self.isWork = True

        # Setting the flag
        self.isUpdate = True if server is not None else False
        self.datamaker = DataMaker('./data/')
        self.detector = HumanDetector()
        self.R_min = RollingAverageFilter(N=10)
        self.R_max = RollingAverageFilter(N=10)
        self.last_stand_time = time.time()
        self.fall_event_count = 0  # 用于计数非连续的跌倒事件
        self.last_human_move = time.time()
        self.last_no_detection = None
        
        # Start socket server
        self.sensor = SenxorConnect()
        # self.server = SocketServer(9753)

        # Setting tracker
        

        self.socket_recv_threading = threading.Thread(target=self.receive_data)
        self.socket_recv_threading.start()
    
    def receive_data(self):
        status = "normal"
        while True:
            new_data = self.sensor.get()  # 62*80
            if new_data is None:
                return
            else:
                self.datamaker.put(new_data)
                T_min = self.R_min(new_data.min()) + 3
                T_max = self.R_max(new_data.max())
                # T_max = 32
                data = np.clip(new_data, a_min=T_min, a_max=T_max)
                data = np.reshape(data, (62, 80))
                
                matplotlib.image.imsave("temp.jpg", data)
                image = cv2.imread('temp.jpg')
                if image is None:
                    print('bad image')
                    continue
                rendered_pic = cv2.resize(image, (400, 310))
                if self.isWork:
                    ret_boxs = yolomodel.inference(rendered_pic, conf_thres=0.2, iou_thres=0.4)
                    # ret_boxs = []
                else:
                    ret_boxs = []
                prob = [row[-1] for row in ret_boxs]
                
                fall_flag = False
                for pred_box in ret_boxs:
                    box = np.squeeze(scale_coords((96, 96), np.expand_dims(pred_box[0:4], axis=0).astype("float"), (310,400)).round(), axis=0).astype("int")
                    cv2.rectangle(rendered_pic, (box[0], box[1]), (box[2], box[3]), self.color[int(pred_box[4])], thickness=3)
                    uid = str(int(pred_box[6]))
                    if uid in name_dic.keys():
                        name = name_dic[uid]
                    else:
                        name = fake.name()
                        name_dic[uid] = name
                    # cv2.putText(rendered_pic, name, (box[0], box[1]-5),cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color[int(pred_box[4])], 2)
                    if pred_box[4] == 1: # and (box[1]+box[3]) < 500:
                        fall_flag = True
                
                cv2.imshow("test", rendered_pic)
                if cv2.waitKey(1) == ord('q'):
                    break
                if cv2.waitKey(1) == ord('r'):
                    self.datamaker.start()
                    print("record start")
                if cv2.waitKey(1) == ord('f'):
                    self.datamaker.end()
                    print("record end")


class SenxorConnect:
    def __init__(self):
        self.mi48, connected_port, port_names = connect_senxor()
        self.mi48.set_fps(10)
        self.mi48.regwrite(0xD0, 0x00)
        self.mi48.disable_filter(f1=True, f2=True, f3=True)
        self.mi48.set_filter_1(10)
        self.mi48.enable_filter(f1=True, f2=True, f3=False, f3_ks_5=False)
        self.mi48.regwrite(0xC2, 0x64)
        # self.mi48.set_emissivity
        # self.mi48.set_offset_corr(0.0)
        # self.mi48.set_sens_factor(100)
        # self.mi48.get_sens_factor()
        self.mi48.start(stream=True, with_header=True)

    def get(self): 
        data, header = self.mi48.read()
        if data is None:
            self.mi48.stop()
            sys.exit(1)
        data = np.array(data)
        return data
    

if __name__ == '__main__':
    onnx_model_path = "Models/onnx/dall.onnx"
    yolomodel = YOLOModel(onnx_model_path)
    rtmpUrl = "rtmp://127.0.0.1:1935/test/aaa"
    serverUrl = "http://127.0.0.1:5000/"
    # s = ThermalServer(time=0.1, push_stream=rtmpUrl, server=serverUrl, model=yolomodel)
    s = ThermalServer(model=yolomodel)
    # s.start_pushing()
