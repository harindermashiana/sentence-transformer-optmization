import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance(models_data):
    """
    Plots the performance of various models using scatter plots for speed vs. accuracy and size comparisons.
    
    Parameters:
    - models_data: Dictionary containing model names and their performance metrics (speed, accuracy, size).
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, metrics in models_data.items():
        # Scatter plot for speed vs. accuracy
        plt.scatter(metrics['speed'], metrics['accuracy'], s=metrics['size']*100, label=model_name)
    
    plt.xlabel('Inference Speed (sec)')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_and_latency(models_data):
    """
    Creates bar charts for comparing accuracy and inference times of different models.
    
    Parameters:
    - models_data: Dictionary containing model names and their performance metrics (accuracy, latency).
    """
    # Data preparation
    names = list(models_data.keys())
    accuracies = [model['accuracy'] for model in models_data.values()]
    latencies = [model['latency'] for model in models_data.values()]

    x = np.arange(len(names))  # Label locations
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for accuracies
    ax1.bar(x - 0.2, accuracies, 0.4, label='Accuracy', color='b')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.tick_params(axis='y', labelcolor='b')

    # Creating a second y-axis to plot latency
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, latencies, 0.4, label='Latency (sec)', color='r')
    ax2.set_ylabel('Latency (sec)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Accuracy and Latency Comparison')
    fig.tight_layout()
    plt.legend()
    plt.show()

def plot_top_categories(categories_data):
    """
    Plots the top categories predicted by each model with their accuracy and inference times.
    
    Parameters:
    - categories_data: Dictionary with model names as keys and another dictionary with categories and their metrics as values.
    """
    fig, axes = plt.subplots(nrows=len(categories_data), figsize=(10, len(categories_data)*3))
    if len(categories_data) == 1:  # Adjust if there's only one model to avoid iteration issues
        axes = [axes]
    
    for ax, (model_name, categories) in zip(axes, categories_data.items()):
        categories_names = list(categories.keys())
        accuracies = [cat['accuracy'] for cat in categories.values()]
        inference_times = [cat['inference_time'] for cat in categories.values()]

        x = np.arange(len(categories_names))
        ax.bar(x - 0.2, accuracies, 0.4, label='Accuracy', color='blue')
        ax.bar(x + 0.2, inference_times, 0.4, label='Inference Time (sec)', color='red')

        ax.set_ylabel('Metrics')
        ax.set_title(f'Top Categories by {model_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(categories_names)
        ax.legend()

    plt.tight_layout()
    plt.show()