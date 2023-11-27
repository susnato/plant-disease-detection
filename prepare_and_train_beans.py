import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, DefaultDataCollator, TrainingArguments, Trainer
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor


beans = load_dataset("beans")
checkpoint = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

id2label = {i:cls for i, cls in enumerate(beans['train'].features['labels'].names)}
label2id = {v:k for k, v in id2label.items()}
model = AutoModelForImageClassification.from_pretrained(checkpoint,
                                                        num_labels=len(label2id.keys()),
                                                        id2label=id2label,
                                                        label2id=label2id,
                                                        ignore_mismatched_sizes=True,
                                                        )


accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

def transforms(examples):
    images = [_transforms(image.convert("RGB")) for image in examples['image']]
    labels = [label for label in examples['labels']]

    return {"pixel_values": torch.stack(images), "label": torch.tensor(labels)}

beans = beans.with_transform(transforms)

data_collator = DefaultDataCollator()


training_args = TrainingArguments(
    output_dir="plant_disease_detection-beans",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    warmup_ratio=0.2,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=beans["train"],
    eval_dataset=beans["validation"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()


