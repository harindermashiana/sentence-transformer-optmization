import wandb
# Disable wandb for this session
wandb.init(mode="disabled")

# Importing necessary modules
from quantization import myquantizer
from pathlib import Path
import datasets
import os
from datasets import load_dataset
from setfit import sample_dataset
from neural_compressor.experimental import Quantization, common

#import custom modules
from student_train import train_student
from teacher_train import train_teacher
from benchmark import ModelEvaluator
from distillation import distill_model
from onnx_model import onnx_conversion
from plot import plot_model_data
from optimum.onnxruntime import ORTModelForFeatureExtraction
from quantization import OnnxSetFitModel

# Disable tokenizer parallelism to avoid multi-threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the AG News dataset
dataset = load_dataset("ag_news")

# Split training dataset for distillation, and sample a subset for the student model
train_dataset = dataset["train"].train_test_split(seed=42)
train_dataset_student = train_dataset["test"].select(range(1000))
train_dataset = sample_dataset(train_dataset["train"])
test_dataset = dataset["test"]

# Train the student model with a smaller transformer model
student_model = train_student("sentence-transformers/paraphrase-MiniLM-L3-v2", train_dataset)

# Evaluate the student model
pb = ModelEvaluator(student_model.model, test_dataset)
b1 = pb.conduct_benchmark()

# Train a teacher model with a larger transformer model for comparison
teacher_model = train_student("sentence-transformers/paraphrase-mpnet-base-v2", train_dataset)
pb = ModelEvaluator(teacher_model.model, test_dataset)
b2 = pb.conduct_benchmark()

# Perform model distillation from teacher to student
distiller = distill_model(student_model.model, teacher_model.model, train_dataset_student)
distiller.student_model._save_pretrained("distilled")

# Evaluate the distilled model
pb = ModelEvaluator(distiller.student_model, test_dataset)
b3 = pb.conduct_benchmark()

# Prepare for ONNX conversion
model_path = "distilled"
ort_model, tokenizer = onnx_conversion(model_path)

# Create a directory for ONNX models
onnx_path = Path("onnx")

# Load the non-quantized ONNX model for feature extraction
ort_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path, file_name="model.onnx")
onnx_setfit_model = OnnxSetFitModel(ort_model, tokenizer, student_model.model.model_head)

# Benchmark the non-quantized ONNX model
b4 = pb.conduct_benchmark_onnx(onnx_setfit_model, "onnx/model.onnx")

# Quantize the ONNX model
quantized_model = myquantizer(onnx_path, student_model.model.model_head, tokenizer, test_dataset)
qm = quantized_model.quantizer_model()

# Load the quantized model for feature extraction
ort_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path, file_name="model_quantized.onnx")
onnx_setfit_model = OnnxSetFitModel(ort_model, tokenizer, student_model.model.model_head)

# Benchmark the quantized ONNX model
b5 = pb.conduct_benchmark_onnx(onnx_setfit_model, "onnx/model_quantized.onnx")


