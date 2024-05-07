from neural_compressor.experimental import Quantization, common

import functools

import evaluate
import onnxruntime
from optimum.onnxruntime import ORTModelForFeatureExtraction
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from setfit.exporters.utils import mean_pooling

accuracy = evaluate.load("accuracy")

class OnnxSetFitModel:
    def __init__(self, ort_model, tokenizer, model_head):
        self.ort_model = ort_model
        self.tokenizer = tokenizer
        self.model_head = model_head

    def predict(self, inputs):
        encoded_inputs = self.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.ort_model(**encoded_inputs)
        embeddings = mean_pooling(
            outputs["last_hidden_state"], encoded_inputs["attention_mask"]
        )
        return self.model_head.predict(embeddings)

    def __call__(self, inputs):
        return self.predict(inputs)

class myquantizer:
  def __init__(self,onnx_path,model_head,tokenizer, test_dataset):
    self.onnx_path = onnx_path
    self.head = model_head
    self.tokenizer = tokenizer
    self.test_dataset = test_dataset

  def eval_func(self, model):
      print(self.onnx_path)
      ort_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_path)
      ort_model.model = onnxruntime.InferenceSession(model.SerializeToString(), None)
      onnx_setfit_model = OnnxSetFitModel(ort_model, self.tokenizer, self.head)
      preds = []
      chunk_size = 100
      for i in tqdm(range(0, len(self.test_dataset["text"]), chunk_size)):
          preds.extend(
              onnx_setfit_model.predict(self.test_dataset["text"][i : i + chunk_size])
          )
      labels = self.test_dataset["label"]
      accuracy_calc = accuracy.compute(predictions=preds, references=labels)
      return accuracy_calc["accuracy"]

  def build_dynamic_quant_yaml(self):
      yaml = """
  model:
    name: bert
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
      with open("build.yaml", "w", encoding="utf-8") as f:
          f.write(yaml)
  def quantizer_model(self):
    self.build_dynamic_quant_yaml()
    onnx_output_path = "onnx/model_quantized.onnx"
    quantizer = Quantization("build.yaml")
    model_is_at = str(self.onnx_path / "model.onnx")
    quantizer.model = common.Model(model_is_at)
    quantizer.eval_func = functools.partial(self.eval_func)
    quantized_model = quantizer()
    quantized_model.save(onnx_output_path)
