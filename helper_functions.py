import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


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
def transform_aug_ann(examples, transform, image_processor):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")

