import numpy as np
import matplotlib.image
import cv2
import sys
sys.path.append("Models")
from models.YOLOnnx import YOLOModel
from utils.preprocess import scale_coords
from utils.detection import LoadData
import time
import os
import shutil
import random
import json
import tifffile


label_dict = {'1': 0, '2': 0, '4': 1, '7': 2}

def convert(img_size, bbox):
    dw = 1. / img_size[0]
    dh = 1. / img_size[1]
    x = (bbox[0][0] + bbox[1][0]) / 2.0
    y = (bbox[0][1] + bbox[1][1]) / 2.0
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    x = x * dw
    y = y * dh
    h = h * dh
    w = w * dw
    return(x, y, w, h)

def norm255(input_matrix):
    mapped_matrix = (input_matrix - np.min(input_matrix)) * (255 / (np.max(input_matrix) - np.min(input_matrix)))
    return mapped_matrix.astype(np.uint8)

def read_pics(loader, subdir, num=3):
    # 随机开始数据帧，并随机跳过几帧
    loader.load(subdir=subdir)
    action = subdir.split('A')[1]
    pics = []
    lens_pic = len(loader)
    if action in ('001', '002', '007', '008'):
        rand_start = random.randint(0, lens_pic-15)
        loader.forward(rand_start)
        
        for i in range(num):
            data = loader.get()
            data = np.clip(data, a_min=18, a_max=30)
            data = np.reshape(data, (62, 80))
            norm_data = norm255(data)
            # if pics == []:
            #     pics = norm_data
            # else:
            #     pics = np.stack((pics, norm_data), axis=-1)
            pics.append(norm_data)
            rand_forward = random.randint(0,2)
            loader.forward(rand_forward)   
        # merged_pic = cv2.merge(pics)
        pics = np.transpose(pics, axes=(1,2,0))
    elif action in ('003'):
        print(pics.shape)
    return pics


def classify_data(data_path, class_path):
    if not os.path.exists(class_path):
        os.mkdir(class_path)
    for data_dir in os.listdir(data_path):  # S001C001P001R001A001
        print(data_dir)
        action_label = data_dir.split("A")[1]
        # if data_dir.split("C")[1][2] != '4':
        #     continue
        if not os.path.exists(os.path.join(class_path, action_label)):
            os.mkdir(os.path.join(class_path, action_label))
        if not os.path.exists(os.path.join(class_path, action_label, data_dir)):
            shutil.copytree(os.path.join(data_path, data_dir), os.path.join(class_path, action_label, data_dir))
    print("-- All data copy to classified dir --")

def npy2channeljpg(class_path, image_path):
    repeat = 10  # 10倍数据
    for label_dir in os.listdir(class_path):
        loader = LoadData(os.path.join(class_path, label_dir))
        if not os.path.exists(os.path.join(image_path, label_dir)):
            os.mkdir(os.path.join(image_path, label_dir))
        for subdir in os.listdir(os.path.join(class_path, label_dir)):
            print(subdir)
            for i in range(repeat):
                merged_image = read_pics(loader, subdir, num=3)
                cv2.imwrite(os.path.join(image_path, label_dir, subdir+'_{}.jpg'.format(i)), merged_image)
            # data = np.load(os.path.join(class_path, label_dir, subdir, filename))
            # matplotlib.image.imsave(os.path.join(image_path, label_dir, subdir+'_'+filename.split('.')[0]+'.jpg'), data)
    print("-- All data transfer to image --")


def npy2channel5jpg(class_path, image_path):
    repeat = 10  # 5倍数据
    for label_dir in os.listdir(class_path):
        loader = LoadData(os.path.join(class_path, label_dir))
        if not os.path.exists(os.path.join(image_path, label_dir)):
            os.mkdir(os.path.join(image_path, label_dir))
        for subdir in os.listdir(os.path.join(class_path, label_dir)):
            print(subdir)
            for i in range(repeat):
                merged_image = read_pics(loader, subdir, num=5)
                tifffile.imsave(os.path.join(image_path, label_dir, subdir+'_{}.tiff'.format(i)), merged_image)
            # data = np.load(os.path.join(class_path, label_dir, subdir, filename))
            # matplotlib.image.imsave(os.path.join(image_path, label_dir, subdir+'_'+filename.split('.')[0]+'.jpg'), data)
    print("-- All data transfer to image --")


def npy2jpg(class_path, image_path):
    for label_dir in os.listdir(class_path):
        if not os.path.exists(os.path.join(image_path, label_dir)):
            os.mkdir(os.path.join(image_path, label_dir))
        for subdir in os.listdir(os.path.join(class_path, label_dir)):
            print(subdir)
            for filename in os.listdir(os.path.join(class_path, label_dir, subdir)):
                try:
                    data = np.load(os.path.join(class_path, label_dir, subdir, filename))
                except Exception as e:
                    print(os.path.join(class_path, label_dir, subdir, filename))
                    continue
                # data = np.clip(data, a_min=18, a_max=30)
                data = np.reshape(data, (62, 80))
                matplotlib.image.imsave(os.path.join(image_path, label_dir, subdir+'_'+filename.split('.')[0]+'.jpg'), data)
    print("-- All data transfer to image --")


def read_npy():
    onnx_model_path = "Models/onnx/toilet3.onnx"
    yolomodel = YOLOModel(onnx_model_path)
    color = [(0,255,0), (0,0,255), (255,0,0)]
    data_path = './data/'
    loader = LoadData(data_path)
    loader.load(subdir='S005C004P005R002A004')
    while True:
        time.sleep(0.1)
        new_data = loader.get()
        # print(loader.src())
        if new_data is None:
            break
        else:
            data = np.clip(new_data, a_min=18, a_max=30)
            data = np.reshape(data, (62, 80))
            
            matplotlib.image.imsave("temp.jpg", data)
            image = cv2.imread('temp.jpg')
            if image is None:
                print('bad image')
                continue
            rendered_pic = cv2.resize(image, (400, 310))
            ret_boxs = yolomodel.inference(rendered_pic)
            # ret_boxs = []
            prob = [(row[4], row[-1]) for row in ret_boxs]
            fall_flag = False
            for pred_box in ret_boxs:
                # print(pred_box[4])
                box = np.squeeze(scale_coords((96, 96), np.expand_dims(pred_box[0:4], axis=0).astype("float"), (310,400)).round(), axis=0).astype("int")
                cv2.rectangle(rendered_pic, (box[0], box[1]), (box[2], box[3]), color[int(pred_box[4])], thickness=3)
                if pred_box[4] == 1:
                    fall_flag = True
            cv2.imshow("test", rendered_pic)
            if cv2.waitKey(1) == ord('q'):
                break

def labelme2yolo(images_class_data, pre_data_path):
    if not os.path.exists(pre_data_path):
        os.mkdir(pre_data_path)
        os.mkdir(os.path.join(pre_data_path, 'labels'))
        os.mkdir(os.path.join(pre_data_path, 'images'))
    
    for labeldir in os.listdir(images_class_data):
        for filename in os.listdir(os.path.join(images_class_data, labeldir)):
            if filename.endswith('.json'):
                json_file_path = os.path.join(images_class_data, labeldir, filename)
                image_file_path = json_file_path.split('.')[0] + '.jpg'
                with open(json_file_path) as jsonf:
                    data = json.load(jsonf)
                    img_h = data["imageHeight"]
                    img_w = data["imageWidth"]
                    shape = data["shapes"][0]
                    thermal_label = shape["label"]
                    thermal_bbox = shape["points"]
                    x, y, w, h = convert([img_w, img_h], thermal_bbox)
                new_txt_name = filename.split('.')[0] + '.txt'
                with open(os.path.join(pre_data_path, 'labels', new_txt_name), 'w') as txtf:
                    txtf.write(str(label_dict[thermal_label]) + " " + " ".join([str(x), str(y), str(w), str(h)]))
                shutil.copy(image_file_path, os.path.join(pre_data_path, 'images'))

def label_jpg(onnx_model_path, image_path, label_path):
    onnx_model_path = "./Models/onnx/toilet3.onnx"
    yolomodel = YOLOModel(onnx_model_path)
    label_dic = {"001":-1, "002":0, "003":-1, "004":1, "005":-1, "006":-1,"007":2, "008":1}
    color = [(0,255,0), (0,0,255), (255,0,0)]
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    for label_dir in os.listdir(image_path):
        if label_dir in label_dic.keys():
            label = label_dic[label_dir]
        else:
            continue
        if label >= 0:
            if not os.path.exists(os.path.join(label_path, label_dir)):
                os.mkdir(os.path.join(label_path, label_dir))
                os.mkdir(os.path.join(label_path, label_dir, 'labels'))
                os.mkdir(os.path.join(label_path, label_dir, 'images'))
            for filename in os.listdir(os.path.join(image_path, label_dir)):
                image = cv2.imread(os.path.join(image_path, label_dir, filename))
                boxs = yolomodel.inference(image)
                if boxs is not []:
                    for pred_box in boxs:
                        box = np.squeeze(scale_coords((96, 96), np.expand_dims(pred_box[0:4], axis=0).astype("float"), (62,80)).round(), axis=0).astype("int")
                        xywh = convert([80, 62], [[box[0], box[1]],[box[2], box[3]]])
                        with open(os.path.join(label_path, label_dir, 'labels', filename.split('.')[0]+'.txt'), 'a') as txtf:
                            txtf.write(str(label) + " " + " ".join([str(xywh[0]), str(xywh[1]), str(xywh[2]), str(xywh[3])]))
                            txtf.write('\n')
                else:
                    print("no detection in {}".format(filename))
                    continue
                shutil.copy(os.path.join(image_path, label_dir, filename), os.path.join(os.path.join(label_path, label_dir, 'images'), filename))
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color[int(pred_box[4])], thickness=3)
                cv2.imshow("test", image)
                if cv2.waitKey(1) == ord('q'):
                    break
        else:
            continue

def label_channeljpg(onnx_model_path, image_path, label_path):
    onnx_model_path = "./Models/onnx/toilet3.onnx"
    yolomodel = YOLOModel(onnx_model_path)
    label_dic = {"001":0, "002":1, "003":2, "004":3, "005":4, "006":5,"007":6, "008":7}
    color = [(0,255,0), (0,0,255), (255,0,0)]
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    for label_dir in os.listdir(image_path):
        if label_dir in label_dic.keys():
            label = label_dic[label_dir]
        else:
            continue
        if label >= 0:
            if not os.path.exists(os.path.join(label_path, label_dir)):
                os.mkdir(os.path.join(label_path, label_dir))
                os.mkdir(os.path.join(label_path, label_dir, 'labels'))
                os.mkdir(os.path.join(label_path, label_dir, 'images'))
            for filename in os.listdir(os.path.join(image_path, label_dir)):
                image = cv2.imread(os.path.join(image_path, label_dir, filename))
                middle_channel = image[:, :, 1]
                matplotlib.image.imsave("temp.jpg", middle_channel)
                image = cv2.imread('temp.jpg')
                boxs = yolomodel.inference(image)
                if boxs is not []:
                    for pred_box in boxs:
                        box = np.squeeze(scale_coords((96, 96), np.expand_dims(pred_box[0:4], axis=0).astype("float"), (62,80)).round(), axis=0).astype("int")
                        xywh = convert([80, 62], [[box[0], box[1]],[box[2], box[3]]])
                        with open(os.path.join(label_path, label_dir, 'labels', filename.split('.')[0]+'.txt'), 'a') as txtf:
                            txtf.write(str(label) + " " + " ".join([str(xywh[0]), str(xywh[1]), str(xywh[2]), str(xywh[3])]))
                            txtf.write('\n')
                else:
                    print("no detection in {}".format(filename))
                    continue
                shutil.copy(os.path.join(image_path, label_dir, filename), os.path.join(os.path.join(label_path, label_dir, 'images'), filename))
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color[int(pred_box[4])%3], thickness=3)
                cv2.imshow("test", image)
                if cv2.waitKey(1) == ord('q'):
                    break
        else:
            continue

def label_5channeljpg(onnx_model_path, image_path, label_path):
    onnx_model_path = "./Models/onnx/toilet3.onnx"
    yolomodel = YOLOModel(onnx_model_path)
    label_dic = {"001":0, "002":1, "003":2, "004":3, "005":4, "006":5,"007":6, "008":7}
    color = [(0,255,0), (0,0,255), (255,0,0)]
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    for label_dir in os.listdir(image_path):
        if label_dir in label_dic.keys():
            label = label_dic[label_dir]
        else:
            continue
        if label >= 0:
            if not os.path.exists(os.path.join(label_path, label_dir)):
                os.mkdir(os.path.join(label_path, label_dir))
                os.mkdir(os.path.join(label_path, label_dir, 'labels'))
                os.mkdir(os.path.join(label_path, label_dir, 'images'))
            for filename in os.listdir(os.path.join(image_path, label_dir)):
                image = tifffile.imread(os.path.join(image_path, label_dir, filename))
                middle_channel = image[:, :, 4]
                matplotlib.image.imsave("temp.jpg", middle_channel)
                image = cv2.imread('temp.jpg')
                boxs = yolomodel.inference(image)
                if boxs != []:
                    for pred_box in boxs:
                        box = np.squeeze(scale_coords((96, 96), np.expand_dims(pred_box[0:4], axis=0).astype("float"), (62,80)).round(), axis=0).astype("int")
                        xywh = convert([80, 62], [[box[0], box[1]],[box[2], box[3]]])
                        with open(os.path.join(label_path, label_dir, 'labels', filename.split('.')[0]+'.txt'), 'a') as txtf:
                            txtf.write(str(label) + " " + " ".join([str(xywh[0]), str(xywh[1]), str(xywh[2]), str(xywh[3])]))
                            txtf.write('\n')
                    shutil.copy(os.path.join(image_path, label_dir, filename), os.path.join(os.path.join(label_path, label_dir, 'images'), filename))
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color[int(pred_box[4])%3], thickness=3)
                    cv2.imshow("test", image)
                else:
                    print("no detection in {}".format(filename))
                    continue

                if cv2.waitKey(1) == ord('q'):
                    break
        else:
            continue

def make_dataset(data_path, dataset_path):
    train_percent = 0.6
    val_percent = 0.2
    test_percent = 0.2
    # data_path = './pre_data/'
    # dataset_path = './toilet_data'

    train_image_path = os.path.join(dataset_path, './train/images/')
    train_label_path = os.path.join(dataset_path, './train/labels/')

    val_image_path = os.path.join(dataset_path, './val/images/')
    val_label_path = os.path.join(dataset_path, './val/labels/')

    test_image_path = os.path.join(dataset_path, './test/images/')
    test_label_path = os.path.join(dataset_path, './test/labels/')

    def gather_data():
        for action_name in os.listdir(data_path):
            for data in os.listdir(os.path.join(data_path, action_name, 'images')):
                shutil.copy(os.path.join(data_path, action_name, 'images', data), os.path.join(dataset_path, 'images', data))
            for data in os.listdir(os.path.join(data_path, action_name, 'labels')):
                shutil.copy(os.path.join(data_path, action_name, 'labels', data), os.path.join(dataset_path, 'labels', data))

    def mkdir():
        if not os.path.exists(train_image_path):
            os.makedirs(train_image_path)
        if not os.path.exists(train_label_path):
            os.makedirs(train_label_path)
        if not os.path.exists(val_image_path):
            os.makedirs(val_image_path)
        if not os.path.exists(val_label_path):
            os.makedirs(val_label_path)
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        if not os.path.exists(test_label_path):
            os.makedirs(test_label_path)

    def split_data(image_path, label_path):
        total_data = os.listdir(label_path)
        num_data = len(total_data)
        data_num_list = range(num_data)

        num_train = int(num_data * train_percent)
        num_val = int(num_data * val_percent)
        num_test = int(num_data * test_percent)

        train = random.sample(data_num_list, num_train)
        remain = [i for i in data_num_list if not i in train]
        val = random.sample(remain, num_val)
        test = [i for i in remain if not i in val]
        num_test = len(test)

        print("train: {}, val: {}, test: {}".format(num_train, num_val, num_test))
        for i in data_num_list:
            record_name = total_data[i].split('.')[0]
            C = record_name.split('C')[1][2]
            if C != '6':
                continue
            src_image = os.path.join(image_path, record_name+'.tiff')
            src_label = os.path.join(label_path, record_name+'.txt')

            if i in train:
                dst_image = os.path.join(train_image_path, record_name+'.tiff')
                dst_label = os.path.join(train_label_path, record_name+'.txt')
            elif i in val:
                dst_image = os.path.join(val_image_path, record_name+'.tiff')
                dst_label = os.path.join(val_label_path, record_name+'.txt')
            elif i in test:
                dst_image = os.path.join(test_image_path, record_name+'.tiff')
                dst_label = os.path.join(test_label_path, record_name+'.txt')
            else:
                print('error: record not assigned')
                continue
            shutil.copy(src_image, dst_image)
            shutil.copy(src_label, dst_label)
    
    # gather_data()
    mkdir()
    split_data(os.path.join(dataset_path, 'images'), os.path.join(dataset_path, 'labels'))
    

if __name__ == "__main__":
    # read_jpg()
    data_path = 'data/mutiperspective_data'  # 原始npy文件夹
    class_path = 'data/classified_data'  # 分类后的npy文件夹
    image_data = 'data/images_class_data'  # 转为图片后的文件夹
    label_data = 'data/labels_data'
    pre_data = 'data/pre_data/'
    dataset_path = 'data/C10_data'
    onnx_model_path = "Models/onnx/toilet3.onnx"
    if not os.path.exists(class_path):
        os.mkdir(class_path)
    if not os.path.exists(image_data):
        os.mkdir(image_data)

    # read_npy()
    # classify_data(data_path, class_path)  # 按照action来对数据进行分类
    # npy2jpg(class_path, image_data)  # 把npy文件转化为对应的jpg
    # npy2channel5jpg(class_path, image_data)
    # npy2channeljpg(class_path, image_data)
    # labelme2yolo(image_data, pre_data)  # 把labelme得到的class-图片+json转化为yolo需要的images+labels格式
    # label_channeljpg(onnx_model_path, image_data, label_data)  # 把jpg文件打上框作为label
    # label_5channeljpg(onnx_model_path, image_data, label_data)
    make_dataset(label_data, dataset_path)  # 把图片和标签数据制作为yolo训练用的数据集
    
