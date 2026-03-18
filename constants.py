DATA_DIR="TrainingNames.txt"
SAVE_DIR="vocab.json"
LR=0.01
EPOCHS=30
def get_model_save_path(datetime):
    return f"models/model_{datetime}"
