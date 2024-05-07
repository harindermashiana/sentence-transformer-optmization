import os
from pathlib import Path
import wandb
from datasets import load_dataset
import json
os.environ['WANDB_API_KEY'] = '3065b922fcb0dd1f483df3a6dfe54f4b93daf086'
# Disable wandb for this session
wandb.init(mode="online")

# Custom module imports
from training.quantization import ModelQuantizer
from training.student_train import train_student_model
from training.teacher_train import train_teacher_model
from evaluator.benchmark import ModelBenchmark  # Updated class name
from training.distillation import perform_model_distillation
from training.onnx_model import convert_to_onnx
from training.quantization import EnhancedOnnxModel

from optimum.onnxruntime import ORTModelForFeatureExtraction
from setfit import sample_dataset

# Setting up environment variable to avoid parallelism issues with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the AG News dataset
dataset = load_dataset("ag_news")

# Prepare datasets for training and evaluation
train_dataset = dataset["train"].train_test_split(seed=42)
train_dataset_student = train_dataset["test"].select(range(1000))
train_dataset = sample_dataset(train_dataset["train"])
test_dataset = dataset["test"]

# Train and evaluate models
wandb.run.name = "Training Student Model"
wandb.init(project="model_analysis", entity="hsmashiana", name="Training Student Model")

student_model = train_student_model("sentence-transformers/paraphrase-MiniLM-L3-v2", train_dataset)
benchmark_evaluator = ModelBenchmark(student_model.model, test_dataset)
student_benchmark = benchmark_evaluator.conduct_benchmark()
wandb.log(student_benchmark)
wandb.log({"train_loss_student": student_model.state.log_history})

wandb.run.name = "Training Teacher Model"
wandb.init(project="model_analysis", entity="hsmashiana", name="Training Teacher Model")

teacher_model = train_teacher_model("sentence-transformers/paraphrase-mpnet-base-v2", train_dataset)
benchmark_evaluator.model = teacher_model.model 
teacher_benchmark = benchmark_evaluator.conduct_benchmark()
wandb.log(teacher_benchmark)
wandb.log({"train_loss_teacher": teacher_model.state.log_history})

# Perform and evaluate model distillation
wandb.run.name = "Distillation"
wandb.init(project="model_analysis", entity="hsmashiana", name="Distillation")

distiller = perform_model_distillation(student_model.model, teacher_model.model, train_dataset_student)
distiller.model.save_pretrained("distilled")
benchmark_evaluator.model = distiller.model  # Update model in the evaluator for re-use
distiller_benchmark = benchmark_evaluator.conduct_benchmark()
wandb.log(distiller_benchmark)
wandb.log({"train_loss_distillation": teacher_model.state.log_history})

# ONNX conversion for the distilled model
wandb.run.name = "Onnx_non_quantized"
wandb.init(project="model_analysis", entity="hsmashiana", name="Onnx_non_quantized")

model_directory = Path("distilled")
converted_model, converted_tokenizer = convert_to_onnx(model_directory)
onnx_setfit_model = EnhancedOnnxModel(converted_model, converted_tokenizer, student_model.model.model_head)

# Benchmarking non-quantized ONNX model
onnx_benchmark_evaluator = ModelBenchmark(onnx_setfit_model, test_dataset)
non_quantized_benchmark = onnx_benchmark_evaluator.conduct_benchmark_onnx(onnx_setfit_model, "onnx/model.onnx")
wandb.log(non_quantized_benchmark)

# Quantize the ONNX model
wandb.run.name = "Onnx_Quantized"
wandb.init(project="model_analysis", entity="hsmashiana", name="Onnx_Quantized")

onnx_quantizer = ModelQuantizer(Path("onnx"), student_model.model.model_head, converted_tokenizer, test_dataset)
quantized_onnx_model = onnx_quantizer.quantize_model()

# Load and benchmark the quantized ONNX model
# Load the quantized model for feature extraction
ort_model = ORTModelForFeatureExtraction.from_pretrained(Path("onnx"), file_name="model_quantized.onnx")
onnx_setfit_model_quantized = EnhancedOnnxModel(ort_model, converted_tokenizer, student_model.model.model_head)

quantized_benchmark_evaluator = ModelBenchmark(onnx_setfit_model_quantized, test_dataset)
quantized_model_benchmark = quantized_benchmark_evaluator.conduct_benchmark_onnx(onnx_setfit_model_quantized, "onnx/model_quantized.onnx")

wandb.run.name = "Onnx_Quantized"
wandb.init(project="model_analysis", entity="hsmashiana", name="Onnx_Quantized")
wandb.log(quantized_model_benchmark)

# Final results output
results = {
    "student_model": student_benchmark,
    "teacher_model": teacher_benchmark,
    "distilled_model": distiller_benchmark,
    "non_quantized_onnx": non_quantized_benchmark,
    "quantized_onnx": quantized_model_benchmark
}

# Optionally, print the results or use them in further analysis
with open('results.json', 'w') as json_file:
    json.dump(results, json_file)