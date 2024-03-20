from utils.detection import LoadData
import numpy as np
import cv2

def pics2channels():
    pass

def norm255(input_matrix):
    mapped_matrix = (input_matrix - np.min(input_matrix)) * (255 / (np.max(input_matrix) - np.min(input_matrix)))
    return mapped_matrix.astype(np.uint8)

def read_pics(loader, subdir, num=10):
    loader.load(subdir=subdir)
    pics = []
    for i in range(num):
        data = loader.get()
        data = np.clip(data, a_min=18, a_max=30)
        data = np.reshape(data, (62, 80))
        norm_data = norm255(data)
        pics.append(norm_data)
        loader.forward(5)
    merged_pic = cv2.merge(pics)
    return merged_pic


if __name__ == '__main__':
    data_path = './data/'
    loader = LoadData(data_path)
    merged_image = read_pics(loader, 'S005C004P008R003A003', num=3)
    cv2.imshow('Merged Image', merged_image)
    cv2.waitKey(0)