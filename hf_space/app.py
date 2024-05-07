import gradio as gr
from transformers import AutoTokenizer
from setfit import SetFitModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
from quantization import OnnxSetFitModel
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load the models
model1 = SetFitModel.from_pretrained("hsmashiana/basemodel_hpml")
ort_model = ORTModelForFeatureExtraction.from_pretrained("hsmashiana/optimized_model_hpml", file_name="model_quantized.onnx")
tokenizer = AutoTokenizer.from_pretrained("hsmashiana/optimized_model_hpml")
model3 = OnnxSetFitModel(ort_model, tokenizer, model1.model_head)

decode = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def plot_throughput_bar_chart(throughput_model1, throughput_model2):
    labels = ['Base model', 'Optimized model']
    throughputs = [throughput_model1, throughput_model2]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, throughputs, color=['blue', 'navy'])
    plt.xlabel('Models')
    plt.ylabel('Throughput (tokens/second)')
    plt.title('Model Throughput Comparison')
    plt.tight_layout()

    # Create a PIL Image from the plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def compare_models(text):
    inputs = tokenizer(text, return_tensors="pt")
    times = []

    # Warm-up phase to ensure fair timing
    for _ in range(5):
        model1([text])

    # Measure the execution time of model predictions
    for _ in range(20):
        start = perf_counter()
        out1 = model1([text])
        end = perf_counter()
        times.append(end - start)

    avg_latency_ms_model_1 = np.mean(times) * 1000

    times = []

    # Warm-up phase to ensure fair timing
    for _ in range(5):
        model3.predict([text])

    # Measure the execution time of model predictions
    for _ in range(20):
        start = perf_counter()
        out3 = model3.predict([text])
        end = perf_counter()
        times.append(end - start)

    avg_latency_ms_model_3 = np.mean(times) * 1000

    throughput_tokens_per_sec1 = inputs['input_ids'].size(1) / (avg_latency_ms_model_1 / 1000)
    throughput_tokens_per_sec2 = inputs['input_ids'].size(1) / (avg_latency_ms_model_3 / 1000)


    plot_data = plot_throughput_bar_chart(throughput_tokens_per_sec1, throughput_tokens_per_sec2)

    result1 = {
        "Base Model": {
            "answer": decode[out1.numpy()[0]],
            "average time (ms)": avg_latency_ms_model_1,
            "throughput (tokens/sec)": throughput_tokens_per_sec1
        }}
    result2 = {
        "Optimized Model": {
            "answer": decode[out3.numpy()[0]],
            "average time (ms)": avg_latency_ms_model_3,
            "throughput (tokens/sec)": throughput_tokens_per_sec2
        }}
    
    return result1, result2, plot_data

iface = gr.Interface(
    fn=compare_models,
    inputs="text",
    outputs=[
        gr.components.JSON(label="Base Model"),
        gr.components.JSON(label="Optimized Model"),
        gr.components.Image(label="throughput Comparison")
    ],
    title="Compare Sentence Classification Models",
    description="Enter a sentence to see how each model classifies it and their throughputs.",
    allow_flagging="never"
)

iface.launch()
