import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

def load_data(json_filename):
    """Load data from a JSON file."""
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def plot_model_performance(accuracy_dict):
    """Plot the model performance as a scatter plot."""
    plt.figure(figsize=(10, 10))
    scatters = []
    color_map = plt.cm.get_cmap('viridis')
    num_models = len(accuracy_dict)
    color_intensity = [color_map(i / num_models) for i in range(num_models-1)]
    color_intensity.append('red')  # Brightest color for 'quantized_onnx'
    
    for idx, (model_name, metrics) in enumerate(accuracy_dict.items()):
        alpha_value = 1 if model_name == "quantized_onnx" else 0.5
        scatter = plt.scatter(metrics['time'], metrics['accuracy'] * 100,
                              s=metrics['size'] * 15, label=model_name,
                              alpha=alpha_value, marker='o', color=color_intensity[idx])
        scatters.append(scatter)
    
    plt.style.use('ggplot')
    plt.xlabel('Inference Speed (sec)')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance Scatter Plot')
    plt.ylim(70, 95)
    plt.xlim(0, 14)
    
    legend_labels = [f'{model_name} (size {metrics["size"]})' for model_name, metrics in accuracy_dict.items()]
    legend_handles = [plt.scatter([], [], s=metrics['size'] * 0.5, color=color_intensity[i]) for i, (scatter, metrics) in enumerate(zip(scatters, accuracy_dict.values()))]
    legend = plt.legend(handles=legend_handles, labels=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title='Model Size')
    for text in legend.get_texts():
        text.set_color('black')

    plt.grid(True)
    plt.savefig('model_performance.png')

def plot_accuracy_comparison(accuracy_dict):
    """Plot the accuracies of the models as a bar chart with values displayed on top of the bars."""
    names = list(accuracy_dict.keys())
    accuracies = [model['accuracy'] * 100 for model in accuracy_dict.values()]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color='#7C7CE5')  # Create bars
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of Model Accuracies')
    plt.ylim(60, 95)
    plt.xticks(rotation=45)
    plt.grid(True)

    # Add text on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, '{:.1f}'.format(yval), ha='center', va='bottom')

    plt.savefig('accuracy_comparison.png')  # Save the figure


def plot_throughput_comparison(accuracy_dict, model_name, text_example):
    """Plot a bar chart comparing throughput in tokens per second."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text_example, return_tensors="pt")
    
    names = list(accuracy_dict.keys())
    latencies = [model['time'] for model in accuracy_dict.values()]
    throughput_tokens_per_sec = [inputs['input_ids'].size(1) / (latency / 1000) for latency in latencies]

    plt.figure(figsize=(10, 6))
    plt.figure(constrained_layout=True)
    plt.bar(names, throughput_tokens_per_sec, color='#7C7CE5')
    plt.xlabel('Models')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Comparison of Model Throughput')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.savefig('model_throughput_comparison.png')

if __name__ == "__main__":
    json_filename = '../results.json'  # Define the JSON file name
    accuracy_dict = load_data(json_filename)
    text_example = "Example text for throughput calculation."
    optimized_model_name = "hsmashiana/optimized_model_hpml"

    plot_model_performance(accuracy_dict)
    plot_accuracy_comparison(accuracy_dict)
    plot_throughput_comparison(accuracy_dict, optimized_model_name, text_example)