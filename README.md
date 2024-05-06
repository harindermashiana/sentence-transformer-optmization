# Optimizing Sentence Transformer Models for Few-Shot Text Classification

## Project Title
Enhancing Efficiency with Knowledge Distillation and Quantization

## Team Members
- Harinder Singh Mashiana (hm3008)
- Mohini Bhave (mb5157)

## Overview
This project aims to enhance the efficiency of Sentence Transformer models by employing advanced techniques such as knowledge distillation and quantization. These methodologies are intended to reduce the model size and improve inference speed, making the models more suitable for deployment in resource-constrained environments. We focus on optimizing these models specifically for few-shot text classification tasks, demonstrating significant improvements in performance metrics.

## Objectives
- **Efficiency Enhancement**: Reduce the computational overhead and model size while retaining high accuracy, especially in few-shot learning scenarios.
- **Deployment Readiness**: Prepare models that can be efficiently deployed in environments with limited resources.
- **Performance Benchmarking**: Evaluate the performance of the models across various metrics, including accuracy, inference speed, and model size.

## Challenges
- **Integration of Techniques**: Seamlessly integrating both knowledge distillation and quantization within the SetFit framework without degrading performance.
- **Balancing Compression and Performance**: Achieving a significant reduction in model size and inference time while maintaining competitive accuracy.

## Approach and Implementation

### Techniques Employed

#### Knowledge Distillation
In our project, knowledge distillation involves a teacher-student model architecture, where a more complex and larger "teacher" model imparts knowledge to a more compact "student" model.

**Methodology**:
1. **Training the Teacher**: Initially, the teacher model is fine-tuned on a specific task to achieve high performance. This model sets the performance benchmark for the student.
2. **Distillation Process**: The student model trains not only to match the final output labels but also to mimic the softer probabilities (softmax outputs) from the teacher model. This is achieved by employing a distillation-specific loss function.
3. **Loss Functions**:
   - **Cross-Entropy Loss**: Targets the accuracy of the student model on the actual task labels.
   - **Distillation Loss (Kullback-Leibler divergence)**: Ensures the student model's output distribution closely approximates that of the teacher model.

#### Quantization
We apply post-training dynamic quantization to the student model, converting numerical parameters from floating-point to lower-bit integers, reducing memory usage and accelerating inference times.

**Methodology**:
1. **Model Conversion**: Convert the trained student model into a format suitable for quantization, such as ONNX.
2. **Quantization Application**: Using tools like ONNX Runtime, apply dynamic quantization to reduce the model's weight precision effectively.

#### Performance Benchmarking
Our project uses a suite of benchmarks to assess the optimized models against traditional models, focusing on text classification tasks.

**Metrics**:
- **Accuracy**: The percentage of correct predictions made by the model.
- **Inference Speed**: The time taken by the model to process inputs and make predictions.
- **Model Size**: The total disk space occupied by the model.

### Implementation Details

#### Model Training
- **SetFitModel**: Adapted Sentence Transformer models configured for SetFit, utilized as both teacher and student in our distillation strategy.
- **Training Scripts**: Custom scripts manage model training, ensuring both teacher and student models are optimally prepared for the distillation process.

#### Model Distillation
- **SetFitDistillationTraining**: A customized training regimen where the student model learns from the teacher through both traditional and distillation loss functions.
- **Implementation**: We enhance the standard training loop to integrate the calculation and application of the distillation loss.

#### ONNX Conversion and Quantization
- **ONNX Conversion**: The trained models are converted to ONNX format to facilitate interoperability and flexibility.
- **Dynamic Quantization**: Post-conversion, the models undergo dynamic quantization to utilize 8-bit integers for weights, reducing their precision but preserving essential performance characteristics.

#### Performance Evaluation
- **Benchmark Suite**: A comprehensive evaluation setup measures accuracy, speed, and size across the original and optimized models, ensuring that the enhancements are beneficial.

These detailed methodologies underscore our commitment to advancing the efficiency of machine learning models through innovative techniques, setting a new standard for model performance in resource-constrained environments.


## Demo
To demonstrate the practical applications and benefits of our optimized models, we have developed an interactive demo hosted on Hugging Face Spaces. This demo visually compares the performance of the standard student model against the optimized quantized, distilled model.

### Interactive Demo Features:
- **Live Performance Metrics**: Users can input a text prompt and see real-time comparisons of inference speed and accuracy between the standard and optimized models.
- **Visualization of Results**: The demo includes graphs that display the accuracy and inference times, providing a clear, visual representation of the benefits of our optimizations.
- **User Interaction**: Users are encouraged to test different text prompts to explore how the models perform with various types of input.

Visit our demo here: [Optimized Models Demo](some_link)

This interactive platform is designed to showcase the efficiency improvements in real-world scenarios, making it easier for stakeholders to evaluate the potential of these optimized models in production environments.

## Conclusion
Our project successfully demonstrates that employing techniques like knowledge distillation and quantization can significantly enhance the efficiency of Sentence Transformer models, particularly for few-shot text classification tasks. Here are the key takeaways from our work:

### Key Outcomes:
- **Reduced Model Size and Faster Inference**: Through quantization and knowledge distillation, we achieved notable reductions in model size and inference time without a substantial compromise in accuracy.
- **Enhanced Deployment Capability**: The optimizations make these models ideal for deployment in environments with stringent resource constraints, such as mobile devices or embedded systems.
- **Scalable and Adaptable Framework**: The methodologies and frameworks developed are versatile and can be adapted to various NLP tasks and different model architectures.

### Future Work:
- **Broadening Dataset Applicability**: Expanding our testing and optimization to include a wider range of datasets and languages to further validate the adaptability of our approach.
- **Exploring Additional Optimization Techniques**: Investigating other advanced techniques such as pruning and more aggressive quantization methods to further enhance model performance.
- **Real-World Deployment Studies**: Conducting extensive deployment studies to assess the real-world impact of these optimizations on various platforms.

Through this project, we have laid a foundation for future explorations into model optimization techniques that do not sacrifice performance for efficiency. Our results encourage the adoption of these optimized models in practical applications, promising significant improvements in operational efficiency and usability in constrained environments.