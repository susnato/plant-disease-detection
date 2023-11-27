import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer, BatchFeature


device = "cuda"
label2id = torch.load("./classes.pth")
id2label = dict((v, k) for k, v in label2id.items())

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
dataset = load_dataset("susnato/plant_disease_detection_processed")
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
model.eval()

training_args = TrainingArguments(
    output_dir="detr-resnet-50_finetuned_plant_disease_detection_processed",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=15,
    fp16=True,
    save_steps=50,
    warmup_ratio=0.2,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    remove_unused_columns=False,
    push_to_hub=True,
)

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(np.array(pixel_values), return_tensors="pt")

    labels = {}
    labels_ = batch[0]["labels"]
    for k, v in labels_.items():
        labels[k] = torch.tensor(v)

    batch = {}
    batch["pixel_values"] = encoding["pixel_values"].type(torch.float16)
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = [labels]
    return batch

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
)

trainer.train()

print("TRAINING FINISHED PUSHING THE FINAL MODEL TO HF HUB...")

trainer.push_to_hub()