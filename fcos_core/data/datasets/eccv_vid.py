import os

import torch
import torch.utils.data
from PIL import Image
import sys
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from fcos_core.structures.bounding_box import BoxList

class EccvDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "person",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "bus",
        "motor",
    )

    def __init__(self,dataset_dir,use_difficult=False,transforms = None):
        index_path = os.path.join(dataset_dir,'index.txt')
        self.dataset_dir = dataset_dir
        with open(index_path) as f:
            self.index = f.readlines()
        self.index = [x.strip("\n") for x in self.index]
        self.index = [x.split(' ') for x in self.index]
        self.keep_difficult = use_difficult
        self.transforms = transforms
        # self._annodir = os.path.join(self.root,'xmlannotations_k')
        # self._imgdir = os.path.join(self.root,'images')
        # self.img_item_name = os.listdir(self._annodir)
        cls = EccvDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        img_path = os.path.join(self.dataset_dir,'sequences',self.index[index][0],self.index[index][1]+'.jpg')
        print(img_path)
        # img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index


    def __len__(self):
        return len(self.img_item_name)

    def get_groundtruth(self, index):
        anno_path = os.path.join(self.dataset_dir,'xmlannotations',self.index[index][0],self.index[index][1]+'.xml')
        anno = ET.parse(anno_path).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        # img_id = self.ids[index]
        img_name = self.img_item_name[index][:-4]+'.jpg'
        anno_name = self.img_item_name[index][:-4] + '.xml'
        anno_path = os.path.join(self._annodir,anno_name)
        anno = ET.parse(anno_path).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return EccvDataset.CLASSES[class_id]
