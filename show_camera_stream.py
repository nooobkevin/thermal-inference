from senxor.mi48 import MI48, format_header, format_framestats
from senxor.utils import (
    data_to_frame, remap, cv_filter, cv_render, RollingAverageFilter, connect_senxor
)
import sys
import numpy as np
import cv2

class SenxorConnect:
    def __init__(self):
        self.mi48, connected_port, port_names = connect_senxor()
        self.mi48.set_fps(10)
        self.mi48.regwrite(0xD0, 0x00)
        self.mi48.disable_filter(f1=True, f2=True, f3=True)
        self.mi48.set_filter_1(10)
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

if __name__ == "__main__":
    # Show frame
    senxor = SenxorConnect()
    while True:
        data = senxor.get()
        if data.any():
            frame = data_to_frame(data, (62, 80))
            frame=frame.astype(np.float32)
            frame=cv2.GaussianBlur(frame, (5,5), 0)
            frame=cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)  # Convert to 8-bit unsigned integer
            cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()