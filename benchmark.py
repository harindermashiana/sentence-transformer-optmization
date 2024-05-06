from tqdm.auto import tqdm
import evaluate
import numpy as np
import torch
from time import perf_counter
from pathlib import Path
from setfit.exporters.utils import mean_pooling

codes = {"1":"World","2":"Sci/Tech","3":"Politics","4":"Business"}
# Load the accuracy metric for evaluation purposes
accuracy_metric = evaluate.load("accuracy")

class ModelEvaluator:
    def __init__(self, model, data):
        self.data = data
        self.model = model

    def evaluate_accuracy(self):
        # Obtain predictions and labels from the dataset
        predictions = self.model.predict(self.data["text"])
        ground_truths = self.data["label"]
        # Calculate the accuracy of predictions
        accuracy_results = accuracy_metric.compute(predictions=predictions, references=ground_truths)['accuracy']
        return accuracy_results

    def measure_model_size(self):
        # Retrieve the model's state dictionary
        model_state = self.model.model_body.state_dict()
        temporary_file = Path("temporary_model.pt")
        torch.save(model_state, temporary_file)
        # Compute the model size in megabytes
        model_size_mb = temporary_file.stat().st_size / (1024 ** 2)
        temporary_file.unlink()  # Remove the file after calculation
        return model_size_mb

    def measure_latency(self, sample_query="Are you testing me?"):
        times = []
        # Warm-up phase to ensure fair timing
        for _ in range(20):
            self.model([sample_query])
        # Measure the execution time of model predictions
        for _ in range(200):
            start = perf_counter()
            self.model([sample_query])
            end = perf_counter()
            times.append(end - start)
        # Calculate mean and standard deviation of latency
        avg_latency_ms = np.mean(times) * 1000
        return avg_latency_ms

    def conduct_benchmark(self):
        results = {}
        results["accuracy"]=self.evaluate_accuracy()
        results["time"]=self.measure_latency()
        results["size"] = self.measure_model_size()

        return results

    def conduct_benchmark_onnx(self,model,modelPath):
        results = {}
        self.model=model
        self.model_path = modelPath
        results["accuracy"]=self.compute_accuracy_onnx(model)
        results["time"]=self.measure_latency()
        results["size"] = self.measure_size_onnx()
        return results

    def compute_accuracy_onnx(self,model):
        preds = []
        chunk_size = 100
        for i in tqdm(range(0, len(self.data["text"]), chunk_size)):
            preds.extend(self.model.predict(self.data["text"][i : i + chunk_size]))
        labels = self.data["label"]
        accuracy_onnx = accuracy_metric.compute(predictions=preds, references=labels)['accuracy']
        return accuracy_onnx

    def measure_size_onnx(self):
        size_mb = Path(self.model_path).stat().st_size / (1024 * 1024)
        print(f"Model size (MB) - {size_mb:.2f}")
        return size_mb

