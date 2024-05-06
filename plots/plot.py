
import matplotlib.pyplot as plt
import numpy as np
import json

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
    plt.savefig("Overall.jpg")

def plot_accuracy_comparison(accuracy_dict):
    """Plot the accuracies of the models as a bar chart."""
    names = list(accuracy_dict.keys())
    accuracies = [model['accuracy'] * 100 for model in accuracy_dict.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(names, accuracies, color='blue')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of Model Accuracies')
    plt.ylim(60, 95)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig("accuracy.jpg")

def plot_accuracy_latency_comparison(accuracy_dict):
    """Plot a dual-axis comparison of accuracy and latency."""
    names = list(accuracy_dict.keys())
    accuracies = [model['accuracy'] * 100 for model in accuracy_dict.values()]
    latencies = [model['time'] for model in accuracy_dict.values()]
    x = np.arange(len(names))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - 0.2, accuracies, 0.4, label='Accuracy (%)', color='red')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy (%)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, latencies, 0.4, label='Latency (sec)', color='blue')
    ax2.set_ylabel('Latency (sec)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title('Dual-axis Comparison of Accuracy and Latency')
    fig.tight_layout()
    plt.legend()
    plt.savefig("latency.jpg")

if __name__ == "__main__":
    json_filename = '../results.json'  # Define the JSON file name
    accuracy_dict = load_data(json_filename)
    plot_model_performance(accuracy_dict)
    plot_accuracy_comparison(accuracy_dict)
    plot_accuracy_latency_comparison(accuracy_dict)
