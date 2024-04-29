import queue
import numpy as np
import os


class LoadData:
    def __init__(self, path=None):
        if path is None:
            self.path = "./data/"
        else:
            self.path = path
        self.files = []
        self.file_index = 0
    def load(self, path=None, subdir=None):
        self.file_index = 0
        if path is None:
            pass
        else:
            self.path = path
        if subdir is None:
            for record_time in os.listdir(self.path):
                frames = os.listdir(os.path.join(self.path, record_time))
                frames.sort(key=lambda x:int(x.split('_')[2][:-4]))
                for frame in frames:
                    self.files.append(record_time+'/'+frame)
        else:
            frames = os.listdir(os.path.join(self.path,subdir))
            frames.sort(key=lambda x:int(x.split('_')[2][:-4]))
            for frame in frames:
                    self.files.append(subdir+'/'+frame)
    def get(self):
        try:
            frame = np.load(os.path.join(self.path, self.files[self.file_index]))
            self.file_index += 1
        except IndexError:
            print("Index reach to {}. Run out of data".format(self.file_index))
            return None
        return frame

    def forward(self, frames):
        if self.file_index + frames <= len(self.files):
            self.file_index += frames
        else:
            self.file_index = len(self.files)-1

    def rewind(self, frames):
        if self.file_index > frames:
            self.file_index -= frames
        else:
            self.file_index = 0

    def src(self):
        return self.path+self.files[self.file_index]
    
    def __len__(self):
        return len(self.files)

class HumanDetector:
    def __init__(self, empty_map=None):
        self.time_window = queue.Queue(75)
        self.last_frame = None
        self.action_map = np.zeros((62,80))
        if empty_map is None:
            self.empty_map = np.zeros((62,80))
            self.time_count = 0
        else:
            self.empty_map = empty_map
            self.time_count = 5000
        pass
    def detect(self, frame):
        # print(frame)
        if self.time_window.empty():
            self.last_frame = frame
            self.time_window.put(frame)
            return False
        changed = frame - self.last_frame
        # print(changed)
        for index, value in np.ndenumerate(changed):
            if value >= 2:
                self.action_map[index] = 75
            else:
                self.action_map[index] = self.action_map[index] - 1 if self.action_map[index] > 0 else 0
        print(np.sum(self.action_map>0))        
        self.last_frame = frame
        if self.time_window.full():
            self.time_window.get()
        self.time_window.put(frame)
        if np.sum(self.action_map>0) > 1000:
            return True

    def long_detect(self, frame):
        if self.time_count == 0:
            self.empty_map = frame
            self.time_count += 1
            return False
        else:
            changed = frame - self.empty_map
            if self.time_count < 5000:
                self.time_count += 1000
            else:
                self.time_count += 1
            self.empty_map = (self.time_count - 1)/self.time_count * self.empty_map + frame / self.time_count
            print(np.sum(changed>=2))
            if np.sum(changed>=2) > 120:
                return True
            else:
                return False

            
    def output(self):
        np.save('empty_map.npy', self.empty_map)


class GMMdetector:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def detect(frame):
        fgmask = fgbg.apply(filtered_frame_resized)
        pass
