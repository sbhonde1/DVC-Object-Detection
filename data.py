import os
import shutil
from typing import Any, Dict
from urllib.parse import urlparse
from urllib.request import urlretrieve
import csv
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor
from PIL import Image


def download_data(download_directory: str, data_config: Dict[str, Any]) -> str:
    if not os.path.exists(download_directory):
        os.makedirs(download_directory, exist_ok=True)
    url = data_config["url"]
    filename = os.path.basename(urlparse(url).path)
    filepath = os.path.join(download_directory, filename)
    if not os.path.exists(filepath):
        urlretrieve(url, filename=filepath)
        shutil.unpack_archive(filepath, download_directory)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform():
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)


# Custom dataset for PennFudan based on:
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
class ObliqueDataset(object):
    def __init__(self, data_root, image_transform=None):
        # self.images_root = os.path.join(data_root, "images")
        # self.labels_root = os.path.join(data_root, "labels")
        included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
        self.images = [os.path.join(data_root, fn) for fn in os.listdir(data_root)
                      if any(fn.endswith(ext) for ext in included_extensions)]
        self.labels = [os.path.join(data_root, fn) for fn in os.listdir(data_root) if fn.endswith("txt")]
        # Image transforms to perform
        # self.image_transform = image_transform
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.images = [os.path.join(self.images_root, f) for f in sorted(os.listdir(self.images_root))]
        # self.labels = [os.path.join(self.labels_root, f) for f in sorted(os.listdir(self.labels_root))]
        self.images = sorted(self.images)
        self.labels = sorted(self.labels)

    @staticmethod
    def calculate_area(boxes):
        """
        Calculate bounding box area.

        :param boxes: Bounding boxes with [left, top, right, bottom]
        :type boxes: torch.Tensor

        :return: area of the bounding boxes
        """
        box_dimension = len(boxes.size())
        if (box_dimension == 1) and (boxes.size()[0] != 0):
            return (boxes[3] - boxes[1] + 1) * (boxes[2] - boxes[0] + 1)
        elif box_dimension == 2:
            return (boxes[:, 3] - boxes[:, 1] + 1) * (boxes[:, 2] - boxes[:, 0] + 1)
        else:
            return torch.tensor([])

    def __getitem__(self, index: int):
        """Obtain the sample with the given index.

        :param index: Index of dataset.
        :type index: int

        :returns: RGB image tensor and dictionary of target description in COCO format
        :rtype: torch.Tensor, dict
        """

        # Load data and bounding box
        image = Image.open(self.images[index])
        box_path = self.labels[index]

        # Output of Dataset must be tensor
        # Assume data is RGB
        image = ToTensor()(image.convert("RGB"))  # (3 x H x W)
        bbox_list = []
        class_list = []
        if os.path.exists(box_path):
            with open(box_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=' ')
                for row in csv_reader:
                    left = int(row[0])  # X0
                    top = int(row[1])  # X1
                    right = int(row[2])  # Y0
                    bottom = int(row[3])  # Y1
                    try:
                        c = int(row[4])  # Class
                    except:
                        # Assume binary class if row[4] does not exist
                        c = 1
                    bbox_list.append([left, top, right, bottom])
                    class_list.append(c)
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)
        # if self.image_transform is not None:
        #     image = self.image_transform(image)
        # Class label
        labels = torch.as_tensor(class_list, dtype=torch.int64)
        image_id = torch.tensor([index])
        # Assume all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        # Add 1 to the area to account for pixels starting at 0,0 and not 1,1
        area = ObliqueDataset.calculate_area(boxes)
        target = {"boxes": boxes,
                  "image_id": image_id,
                  "area": area,
                  "labels": labels,
                  "iscrowd": iscrowd}
        return image, target

    def __len__(self):
        return len(self.images)
