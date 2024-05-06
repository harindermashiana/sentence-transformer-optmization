# Import necessary libraries for model conversion and feature extraction
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

def convert_to_onnx(model_identifier):
    """
    Converts a given model to the ONNX format and saves the ONNX model and tokenizer
    in a specified directory for later use.

    Parameters:
    - model_identifier: The identifier of the model to be converted.

    Returns:
    - Tuple containing the converted ONNX model and tokenizer.
    """
    
    # Define the path where the ONNX models will be stored
    onnx_directory = Path("onnx")
    
    # Load the model for feature extraction using the Optimum library
    ort_feature_extractor = ORTModelForFeatureExtraction.from_pretrained(
        model_identifier, from_transformers=True
    )
    
    # Load the tokenizer associated with the model
    model_tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    
    # Save the ORT feature extractor and tokenizer to the specified directory
    ort_feature_extractor.save_pretrained(onnx_directory)
    model_tokenizer.save_pretrained(onnx_directory)
    
    # Return the ONNX model and tokenizer as a tuple
    return ort_feature_extractor, model_tokenizer








