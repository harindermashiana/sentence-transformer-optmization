import os
from pathlib import Path
import wandb
from datasets import load_dataset

# Disable wandb for this session
wandb.init(mode="disabled")

# Custom module imports
from model.quantization import ModelQuantizer
from model.student_train import train_student_model
from model.teacher_train import train_teacher_model
from evaluation.benchmark import ModelBenchmark  # Updated class name
from model.distillation import perform_model_distillation
from model.onnx_model import convert_to_onnx
from plots.plot import plot_model_performance, plot_accuracy_and_latency, plot_top_categories  # Updated for new plotting functions
from optimum.onnxruntime import ORTModelForFeatureExtraction
from quantization import EnhancedOnnxModel

# Setting up environment variable to avoid parallelism issues with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the AG News dataset
ag_news_dataset = load_dataset("ag_news")

# Prepare datasets for training and evaluation
training_data = ag_news_dataset["train"].train_test_split(seed=42)
sampled_train_data = training_data["train"]
student_training_subset = training_data["test"].select(range(1000))
test_data = ag_news_dataset["test"]

# Train models
student_model = train_student_model("sentence-transformers/paraphrase-MiniLM-L3-v2", sampled_train_data)
teacher_model = train_teacher_model("sentence-transformers/paraphrase-mpnet-base-v2", sampled_train_data)

# Evaluate both models using the updated benchmark class
benchmark_evaluator = ModelBenchmark(student_model.model, test_data)
student_benchmark = benchmark_evaluator.run_benchmark()
benchmark_evaluator.model = teacher_model.model  # Update model in the evaluator for re-use
teacher_benchmark = benchmark_evaluator.run_benchmark()

# Perform and evaluate model distillation
distiller = perform_model_distillation(student_model.model, teacher_model.model, student_training_subset)
distiller.model.save_pretrained("distilled")
benchmark_evaluator.model = distiller.model  # Update model in the evaluator for re-use
distiller_benchmark = benchmark_evaluator.run_benchmark()

# ONNX conversion for the distilled model
model_directory = Path("distilled")
converted_model, converted_tokenizer = convert_to_onnx(model_directory)
onnx_setfit_model = EnhancedOnnxModel(converted_model, converted_tokenizer, student_model.model.model_head)

# Benchmarking non-quantized ONNX model
onnx_benchmark_evaluator = ModelBenchmark(converted_model.model, test_data)
non_quantized_benchmark = onnx_benchmark_evaluator.run_benchmark_onnx(onnx_setfit_model, "onnx/model.onnx")

# Quantize the ONNX model
onnx_quantizer = ModelQuantizer(Path("onnx"), student_model.model.model_head, converted_tokenizer, test_data)
quantized_onnx_model = onnx_quantizer.quantize_model()

# Load and benchmark the quantized ONNX model
# Load the quantized model for feature extraction
ort_model = ORTModelForFeatureExtraction.from_pretrained(Path("onnx"), file_name="model_quantized.onnx")
onnx_setfit_model_quantized = EnhancedOnnxModel(ort_model, converted_tokenizer, student_model.model.model_head)

quantized_benchmark_evaluator = ModelBenchmark(onnx_setfit_model_quantized, test_data)
quantized_model_benchmark = onnx_benchmark_evaluator.run_benchmark_onnx(onnx_setfit_model, "onnx/model_quantized.onnx")

# Final results output
results = {
    "student_model": student_benchmark,
    "teacher_model": teacher_benchmark,
    "distilled_model": distiller_benchmark,
    "non_quantized_onnx": non_quantized_benchmark,
    "quantized_onnx": quantized_model_benchmark
}

# Optionally, print the results or use them in further analysis
print(results)