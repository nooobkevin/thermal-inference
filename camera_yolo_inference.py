from senxor.mi48 import MI48, format_header, format_framestats
from senxor.utils import (
    data_to_frame, remap, cv_filter, cv_render, RollingAverageFilter, connect_senxor
)
import sys
import numpy as np
import cv2

model_path='/Users/nooobkevin/Projects/thingx/thermal-inference/yolo_models/epoch20.onnx'
CLASSES={
  0: "60deg_standing",
  1: "60deg_walking",
  2: "60deg_felldown",
  3: "60deg_sitting",
  4: "60deg_squatting",
  5: "50deg_standing",
  6: "50deg_walking",
  7: "50deg_felldown",
  8: "50deg_sitting",
  9: "50deg_squatting",
  10: "40deg_standing",
  11: "40deg_walking",
  12: "40deg_felldown",
  13: "40deg_sitting",
  14: "40deg_squatting",
  15: "30deg_standing",
  16: "30deg_walking",
  17: "30deg_felldown",
  18: "30deg_sitting",
  19: "30deg_squatting"
}

ACTUAL_ANGLE=[
            #   "30deg",
            #   "40deg",
              "50deg",
            #   '60deg',
            ]

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
class SenxorConnect:
    def __init__(self):
        self.mi48, connected_port, port_names = connect_senxor()
        self.mi48.set_fps(10)
        self.mi48.regwrite(0xD0, 0x00)
        self.mi48.disable_filter(f1=True, f2=True, f3=True)
        self.mi48.set_filter_1(0xa)
        self.mi48.enable_filter(f1=True, f2=True, f3=False, f3_ks_5=False)
        self.mi48.regwrite(0xC2, 0x64)
        self.mi48.start(stream=True, with_header=True)

    def get(self): 
        data, header = self.mi48.read()
        if data is None:
            self.mi48.stop()
            sys.exit(1)
        data = np.array(data)
        return data

def data_to_frame(data, shape):
    try:
        frame = np.reshape(data, shape)
        return frame
    except Exception as e:
        print("Error in data_to_frame:", e)
        return None

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    # cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(img, label, ((x_plus_w-x)//2 , (y_plus_h+y)//2 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def load_model(path:str="yolov5s.onnx"):
    model = cv2.dnn.readNetFromONNX(path)
    return model

if __name__ == "__main__":
    # Show frame
    senxor = SenxorConnect()
    model=load_model(model_path)
    scale=1.0
    while True:
        data = senxor.get()
        if not data.any():
            continue
        frame = data_to_frame(data, (62, 80))
        frame=frame.astype(np.float32)
        frame=cv2.GaussianBlur(frame, (5,5), 0)
        frame=cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame=cv2.resize(frame, (224,224), interpolation=cv2.INTER_LINEAR)
        frame = frame.astype(np.uint8)  # Convert to 8-bit unsigned integer
        frame = cv2.flip(frame, 1)
        blob = cv2.dnn.blobFromImage(frame, 
                                     scalefactor=1 / 255, 
                                     size=(224, 224), 
                                     swapRB=False)
        model.setInput(blob)
        outputs = model.forward()
        
        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]
        
        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.05:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    
        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            # If the "xxdeg" in class_name is in any one of ACTUAL_ANGLE, draw bounding box
            if any(angle in detection["class_name"] for angle in ACTUAL_ANGLE):
                draw_bounding_box(
                    frame,
                    class_ids[index],
                    scores[index],
                    round(box[0] * scale),
                    round(box[1] * scale),
                    round((box[0] + box[2]) * scale),
                    round((box[1] + box[3]) * scale),
                )
            
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    senxor.mi48.stop()
    cv2.destroyAllWindows()