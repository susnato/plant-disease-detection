import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def rename_files(labels):
    labels.filename = labels.filename.replace('NCLB.jpg', 'Corn-NCLB.jpg')
    labels.filename = labels.filename.replace('early-blight-1.jpg', 'tomato-early-blight-1.jpg')
    return labels


def add_coco_annot(labels):
    image_ids = dict(
        (k, v) for k, v in zip(labels["filename"].unique(), range(len(labels["filename"].unique()))))
    labels['image_id'] = labels["filename"].map(lambda x: image_ids[x])

    class_labels = dict((k, v) for k, v in zip(labels["class"].unique(), range(len(labels["class"].unique()))))
    labels['class_label'] = labels["class"].map(lambda x: class_labels[x])

    labels.loc[:, "image_height"] = labels["height"]
    labels.loc[:, "image_width"] = labels["width"]

    labels.loc[:, "width"] = labels["xmax"] - labels["xmin"]
    labels.loc[:, "height"] = labels["ymax"] - labels["ymin"]

    labels = labels.drop(["xmax", "ymax"], axis=1)

    return labels[['image_id', 'filename', 'image_height', 'image_width', 'class', 'class_label', 'xmin', 'ymin', 'width', 'height']]

def draw_bbox(image, annots):
    draw = ImageDraw.Draw(image)

    for annot in annots:
        cls, xmin, ymin, width, height = annot
        draw.rectangle((xmin, ymin, xmin + width, ymin + height), outline="red", width=1)
        draw.text((xmin, ymin), "Tomato leaf mosaic virus", fill="white")

    return image


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


# transforming a batch


