import onnxruntime
import onnx
import cv2
import numpy as np
import torch
from utils.preprocess import letterbox, scale_coords, non_max_suppression


class YOLOModel:
    def __init__(self, model_path, targets_num=5):
        self.model = model_path
        self.session = onnxruntime.InferenceSession(model_path)
        self.targets_num = targets_num
        pass
    
    def inference(self, img):
        im = letterbox(img)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        # cv2.imwrite('temp1.jpg', cv2.resize(img, (96, 96)))
        # im = torch.from_numpy(im)
        # im = im.float()  # uint8 to fp16/32
        im = im.astype(np.float32)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        # im = im.astype(float32)
        if len(im.shape) == 3:
            im = im[None]
        ort_inputs = {self.session.get_inputs()[0].name:im}
        ort_outs = self.session.run(None, ort_inputs)[0]
        ort_outs = torch.tensor(ort_outs)
        ort_outs = non_max_suppression(ort_outs, conf_thres=0.2, iou_thres=0.4, classes=None, agnostic=False, max_det=self.targets_num)
        # ort_outs = np.squeeze(ort_outs, axis=0)
        detections = ort_outs[0].numpy()
        # print(ort_outs)
        boxs = []
        for detection in detections:
            classID = int(detection[5])
            confidence = detection[4]
            # print(classID)
            # print(detection[0:4])
            # print(confidence)
            # x ,y, width, height = detection[0:4]
            # x1 = int(x - width/2)
            # y1 = int(y - height/2)
            # x2 = int(x + width/2)
            # y2 = int(y + height/2)
            x1 ,y1, x2, y2 = detection[0:4]
            boxs.append([x1, y1, x2, y2, classID, confidence])
        return boxs
        
if __name__ == '__main__':
    color = [(0,255,0), (0,0,255)]
    onnx_model = "thermal-exp14-best.onnx"
    yolomodel = YOLOModel(onnx_model)

    video = cv2.VideoCapture("two-people.mp4")
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        ret_boxs = yolomodel.inference(frame)
        for pred_box in ret_boxs:
            box = np.squeeze(scale_coords((96, 96), np.expand_dims(pred_box[0:4], axis=0).astype("float"), (310,400)).round(), axis=0).astype("int")
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color[pred_box[4]])
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destoryAllwindows()