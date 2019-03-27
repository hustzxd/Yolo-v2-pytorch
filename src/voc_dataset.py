"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from src.data_augmentation import *

import ipdb


class VOCDataset(Dataset):
    def __init__(self, root_path="data/VOCdevkit", year="2007", mode="train", image_size=448, is_training=True):
        self.data_path = None
        self.data_paths = None
        if (mode in ["train", "val", "trainval", "test"] and year == "2007") or \
                (mode in ["train", "val", "trainval"] and year == "2012"):
            self.data_path = os.path.join(root_path, "VOC{}".format(year))
        elif mode in ["train", "val", "trainval"] and year == "0712":
            self.data_paths = []
            for year in ["2007", "2012"]:
                self.data_paths.append(os.path.join(root_path, "VOC{}".format(year)))
        id_list_path = None
        id_list_paths = None
        if self.data_path is not None:
            id_list_path = os.path.join(self.data_path, "ImageSets/Main/{}.txt".format(mode))
        if self.data_paths is not None:
            id_list_paths = []
            for p in self.data_paths:
                id_list_paths.append(os.path.join(p, "ImageSets/Main/{}.txt".format(mode)))
        if id_list_path is not None:
            self.ids = [id.strip() for id in open(id_list_path)]
        if id_list_paths is not None:
            self.ids = []
            self.divide = []
            for p in id_list_paths:
                ids_07 = [id.strip() for id in open(p)]
                self.divide.append(len(ids_07))
                self.ids.extend(ids_07)
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.ids)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        id = self.ids[item]
        image_path = None
        if self.data_path:
            image_path = os.path.join(self.data_path, "JPEGImages", "{}.jpg".format(id))
        elif self.data_paths:
            if item >= self.divide[0]:
                image_path = os.path.join(self.data_paths[1], "JPEGImages", "{}.jpg".format(id))
            else:
                image_path = os.path.join(self.data_paths[0], "JPEGImages", "{}.jpg".format(id))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_xml_path = None
        if self.data_path:
            image_xml_path = os.path.join(self.data_path, "Annotations", "{}.xml".format(id))
        elif self.data_paths:
            if item >= self.divide[0]:
                image_xml_path = os.path.join(self.data_paths[1], "Annotations", "{}.xml".format(id))
            else:
                image_xml_path = os.path.join(self.data_paths[0], "Annotations", "{}.xml".format(id))

        annot = ET.parse(image_xml_path)

        objects = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            label = self.classes.index(obj.find('name').text.lower().strip())
            objects.append([xmin, ymin, xmax, ymax, label])
        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])
        image, objects = transformations((image, objects))

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
