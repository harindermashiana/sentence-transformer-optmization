# Importing necessary classes from setfit library
from setfit import SetFitModel, SetFitTrainer

# Function to train a student model with provided model name and dataset
def train_student_model(model_identifier, training_data):
    # Load the pre-trained SetFitModel
    initialized_model = SetFitModel.from_pretrained(model_identifier)
    
    # Create a trainer for the model using the specified dataset
    model_trainer = SetFitTrainer(
        model=initialized_model, train_dataset=training_data
    )
    
    # Start the training process
    model_trainer.train()
    
    # Return the trained model trainer object
    return model_trainer