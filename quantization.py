from neural_compressor.experimental import Quantization, common
import functools
import evaluate
import onnxruntime
from optimum.onnxruntime import ORTModelForFeatureExtraction
from tqdm import tqdm
from setfit.exporters.utils import mean_pooling

# Load the accuracy evaluation tool from the evaluate library
model_accuracy = evaluate.load("accuracy")

class EnhancedOnnxModel:
    def __init__(self, model, tokenizer, model_head):
        self.model = model
        self.tokenizer = tokenizer
        self.model_head = model_head

    def predict(self, text_samples):
        # Encode the input text for model processing
        encoded_text = self.tokenizer(
            text_samples, padding=True, truncation=True, return_tensors="pt"
        )
        model_output = self.model(**encoded_text)
        # Apply mean pooling on the output states with attention masks
        pooled_output = mean_pooling(
            model_output["last_hidden_state"], encoded_text["attention_mask"]
        )
        return self.model_head.predict(pooled_output)

    def __call__(self, text_samples):
        return self.predict(text_samples)

class ModelQuantizer:
    def __init__(self, model_path, head, tokenizer, dataset):
        self.model_path = model_path
        self.head = head
        self.tokenizer = tokenizer
        self.dataset = dataset

    def evaluate_model(self, serialized_model):
        print(f"Quantizing model at: {self.model_path}")
        loaded_model = ORTModelForFeatureExtraction.from_pretrained(self.model_path)
        loaded_model.model = onnxruntime.InferenceSession(serialized_model.SerializeToString(), None)
        onnx_model = EnhancedOnnxModel(loaded_model, self.tokenizer, self.head)
        
        predictions = []
        for start in tqdm(range(0, len(self.dataset["text"]), 100)):
            predictions.extend(
                onnx_model.predict(self.dataset["text"][start:start + 100])
            )
        
        computed_accuracy = model_accuracy.compute(predictions=predictions, references=self.dataset["label"])
        return computed_accuracy["accuracy"]

    def create_quantization_config(self):
        config_contents = """
        model:
          name: sentence_transformer
          framework: onnxrt_integerops

        device: cpu

        quantization:
          approach: post_training_dynamic_quant

        tuning:
          accuracy_criterion:
            relative: 0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
        with open("build.yaml", "w", encoding="utf-8") as file:
            file.write(config_contents)

    def quantize_model(self):
        self.create_quantization_config()
        quantization_output_path = "onnx/model_quantized.onnx"
        quantizer = Quantization("build.yaml")
        model_file_path = str(self.model_path / "model.onnx")
        quantizer.model = common.Model(model_file_path)
        quantizer.eval_func = functools.partial(self.evaluate_model)
        quantized_output = quantizer()
        quantized_output.save(quantization_output_path)