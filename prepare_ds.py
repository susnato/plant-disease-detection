import os
import gc
import numpy as np
import pandas as pd

import albumentations
from transformers import AutoImageProcessor

from datasets import DatasetDict, Dataset, Image
from helper_functions import add_coco_annot, formatted_anns, rename_files
from config import DATASET_DIR, remove_classes


train_folder = os.path.join(DATASET_DIR, "TRAIN")
test_folder = os.path.join(DATASET_DIR, "TEST")
train_labels = pd.read_csv(os.path.join(DATASET_DIR, "train_labels.csv"))
test_labels = pd.read_csv(os.path.join(DATASET_DIR, "test_labels.csv"))

train_labels = rename_files(train_labels)
test_labels = rename_files(test_labels)

cls_ = set(train_labels["class"]) - set(remove_classes)

train_labels = train_labels[train_labels["class"].isin(cls_)]
test_labels = test_labels[test_labels["class"].isin(cls_)]

print("SHAPES - ")
print(len(os.listdir(train_folder)), len(os.listdir(test_folder)))
print(train_labels.shape, test_labels.shape)


train_labels = add_coco_annot(train_labels)
test_labels = add_coco_annot(test_labels)

# remove invalid images
print("Dropping Indices - ", train_labels[train_labels.filename.isin(["IMG_1526.jpg", "Early-blight.jpg", "tomato_plants_1_original.JPG?1407178095.jpg",
                                                                      "IMG_2348.jpg", "tomato-septoria-3.jpg", "powdery-mildew-on-squash-leaves.jpg",
                                                                      "tomato-leaves-17427786.jpg"])].index)
train_labels = train_labels.drop(train_labels[train_labels.filename.isin(["IMG_1526.jpg", "Early-blight.jpg", "tomato_plants_1_original.JPG?1407178095.jpg",
                                                                          "IMG_2348.jpg", "tomato-septoria-3.jpg", "tomato-leaves-17427786.jpg",
                                                                          "powdery-mildew-on-squash-leaves.jpg"])].index, axis=0)
print(train_labels.shape, test_labels.shape)

print("------------PREPARING INITIAL DATASET------------")
print("PREPARING TRAINING SPLIT...")

train_filenames = train_labels.filename.unique()

image_ids, images, widths, heights, objects, categories = [], [], [], [], [], []
for filename in train_filenames:
    image_id = train_labels[train_labels.filename == filename]["image_id"].values[0]
    image = train_folder + f"/{filename}"
    width = train_labels[train_labels.filename == filename]["image_width"].values[0]
    height = train_labels[train_labels.filename == filename]["image_height"].values[0]

    areas = np.array(
        train_labels[train_labels.filename == filename]['width'] * train_labels[train_labels.filename == filename][
            'height']).tolist()
    bboxes = train_labels[train_labels.filename == filename][['xmin', 'ymin', 'width', 'height']].values.tolist()
    category = train_labels[train_labels.filename == filename]['class_label'].values.tolist()
    object = {'area': areas, 'bbox': bboxes, 'category': category}

    image_ids.append(image_id)
    images.append(image)
    widths.append(width)
    heights.append(height)
    objects.append(object)

train_ds = Dataset.from_dict(
    {"image_id": image_ids, "image": images, "width": widths, "height": heights, "objects": objects})
train_ds = train_ds.cast_column("image", Image())

print("PREPARING TEST SPLIT...")

test_filenames = test_labels.filename.unique()

image_ids, images, widths, heights, objects, categories = [], [], [], [], [], []
for filename in test_filenames:
    image_id = test_labels[test_labels.filename == filename]["image_id"].values[0]
    image = test_folder + f"/{filename}"
    width = test_labels[test_labels.filename == filename]["image_width"].values[0]
    height = test_labels[test_labels.filename == filename]["image_height"].values[0]

    areas = np.array(
        test_labels[test_labels.filename == filename]['width'] * test_labels[test_labels.filename == filename][
            'height']).tolist()
    bboxes = test_labels[test_labels.filename == filename][['xmin', 'ymin', 'width', 'height']].values.tolist()
    category = test_labels[test_labels.filename == filename]['class_label'].values.tolist()
    object = {'area': areas, 'bbox': bboxes, 'category': category}

    image_ids.append(image_id)
    images.append(image)
    widths.append(width)
    heights.append(height)
    objects.append(object)

test_ds = Dataset.from_dict(
    {"image_id": image_ids, "image": images, "width": widths, "height": heights, "objects": objects})
test_ds = test_ds.cast_column("image", Image())

ds = DatasetDict({"train": train_ds, "test": test_ds})

del train_ds, test_ds
gc.collect()

print(ds["train"].shape, ds["test"].shape)

print("DONE")

print("------------STARTING PREPROCESSING------------")


checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

def transform_aug_ann(examples):
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

ds["train"] = ds["train"].map(transform_aug_ann, batched=True, batch_size=1, num_proc=1)
ds["test"] = ds["test"].map(transform_aug_ann, batched=True, batch_size=1, num_proc=1)

print("DONE")

print("UPLOADING TO HUB...")

ds.push_to_hub(repo_id="susnato/plant_disease_detection_processed")

print("DONE")