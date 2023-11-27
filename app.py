from datasets import load_dataset

import torch
import gradio as gr
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

dataset = load_dataset("beans")

extractor = AutoFeatureExtractor.from_pretrained("susnato/plant_disease_detection-beans")
model = AutoModelForImageClassification.from_pretrained("susnato/plant_disease_detection-beans")

labels = ['angular_leaf_spot', 'rust', 'healthy']

def classify(im):
  features = extractor(im, return_tensors='pt')
  logits = model(features["pixel_values"])[-1]
  probability = torch.nn.functional.softmax(logits, dim=-1)
  probs = probability[0].detach().numpy()
  confidences = {label: float(probs[i]) for i, label in enumerate(labels)}
  return confidences


block = gr.Blocks(theme="gary109/HaleyCH_Theme")
with block:
    gr.HTML(
        """
        <h1 align="center">PLANT DISEASE DETECTION<h1>
        """
    )
    with gr.Group():
        with gr.Row():
            gr.HTML(
                """
                <p style="color:black">
                    <h4 style="font-color:powderblue;">
                        <center>Crop diseases are a major threat to food security, but their rapid identification remains difficult in many parts of the world due to the lack of the necessary infrastructure. The combination of increasing global smartphone penetration and recent advances in computer vision made possible by deep learning has paved the way for smartphone-assisted disease diagnosis.</center>
                        <br>
                        <center>Using A.I. models in plant disease detection and diagnosis has the potential to revolutionize the way we approach agriculture. By providing real-time monitoring and accurate detection of plant diseases, A.I. can help farmers reduce costs and increase crop</center>
                    </h4>
                </p>
                
                <p align="center">
                    <img src="https://huggingface.co/datasets/susnato/stock_images/resolve/main/merged.png">
                </p>
                """
            )

        with gr.Row():
            gr.HTML(
                """
                <p align="center">
                <h4>
                    Our Approach
                </h4>
                    <img src="https://huggingface.co/datasets/susnato/stock_images/resolve/main/diagram.jpeg">
                </p>
                """
            )

    with gr.Group():
        image = gr.Image(type='pil')
        outputs = gr.Label()
        button = gr.Button("Classify")

        button.click(classify,
                     inputs=[image],
                     outputs=[outputs],
                     )

    with gr.Group():
        gr.Examples([
            ["ex1.jpg", "ex3.jpg"],
        ],
            fn=classify,
            inputs=[image],
            outputs=[outputs],
            cache_examples=True
        )

block.launch(debug=False, share=False)