"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import argparse
import shutil
import cv2
import numpy as np
from src.utils import *
import pickle
from src.yolo_net import Yolo

import ipdb

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.01)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--test_set", type=str, default="test",
                        help="For both VOC2007 and 2012, you could choose 3 different datasets: "
                             "train, trainval and val. Additionally, for VOC2007, "
                             "you could also pick the dataset name test")
    parser.add_argument("--year", type=str, default="2007", help="The year of dataset (2007 or 2012)")
    parser.add_argument("--data_path", type=str, default="data/VOCdevkit", help="the root folder of dataset")
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="params")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/only_params_trained_yolo_voc.pth")
    parser.add_argument("--output", type=str, default="predictions")

    args = parser.parse_args()
    return args


def test(opt):
    input_list_path = os.path.join(opt.data_path, "VOC{}".format(opt.year),
                                   "ImageSets/Main/{}.txt".format(opt.test_set))
    image_ids = [id.strip() for id in open(input_list_path)]
    output_folder = os.path.join(opt.output, "VOC{}_{}".format(opt.year, opt.test_set))
    colors = pickle.load(open("src/pallete", "rb"))
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path)
        else:
            model = Yolo(20)
            model.load_state_dict(torch.load(opt.pre_trained_model_path))
            model.cuda()
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
        else:
            model = Yolo(20)
            model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage))
            model.cuda()
    model.eval()

    results_dir = os.path.join(opt.output, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    fps = []
    for cls in CLASSES:
        txt_path = os.path.join(results_dir, 'comp4_det_test_{}.txt'.format(cls))
        fp = open(txt_path, 'w')
        fps.append(fp)

    for id in image_ids:
        image_path = os.path.join(opt.data_path, "VOC{}".format(opt.year), "JPEGImages", "{}.jpg".format(id))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        image = image[None, :, :, :]
        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height
        data = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            logits = model(data)
            predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                          opt.nms_threshold)
        if len(predictions) == 0:
            continue
        else:
            predictions = predictions[0]
        for pred in predictions:
            xmin = int(max(pred[0] / width_ratio, 0))
            ymin = int(max(pred[1] / height_ratio, 0))
            xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
            ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
            print("Object: {}, Bounding box: ({},{}) ({},{})".format(pred[5], xmin, xmax, ymin, ymax))
            fps[CLASSES.index(pred[5])].write(
                '{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(id, pred[4], xmin, ymin, xmax, ymax))
    for fp in fps:
        fp.close()


if __name__ == "__main__":
    opt = get_args()
    test(opt)
