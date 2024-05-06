
from setfit import SetFitModel, SetFitTrainer

# Function to train a teacher model with the specified model name
def train_teacher_model(model_identifier, teacher_dataset):
    # Load the pre-trained SetFitModel
    teacher_model = SetFitModel.from_pretrained(model_identifier)
    
    # Configure the trainer for the teacher model with provided dataset
    teacher_model_trainer = SetFitTrainer(
        model=teacher_model, train_dataset=teacher_dataset
    )
    
    # Execute the training routine
    teacher_model_trainer.train()
    
    # Return the trained model trainer object
    return teacher_model_trainer