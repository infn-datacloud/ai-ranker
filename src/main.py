import os
from training.main import run as training_run
# from inference.main import run as inference_run


if __name__ == "__main__":
    if os.environ.get("TRAINING", False):
        training_run()
    # elif os.environ.get("INFERENCE", False):
    #     inference_run()
    print("Neither TRAINING or INFERENCE env var has been defined")