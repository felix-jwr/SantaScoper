import argparse
from ultralytics import YOLO


def get_args_as_kwargs():
    parser = argparse.ArgumentParser(prog="SantaScoper", description="Automatic Santa Object Detection")

    parser.add_argument("--project", type=str, default="model", help="Name of the project dir to save training outputs.")
    parser.add_argument("--name", type=str, default="SantaScoper", help="Name of the training run, creates subdir within the project folder for logs and outputs.")
    parser.add_argument("--model", type=str, default="../model/yolo11n.pt", help="The path to a pre-trained model. Defaults to downloading YOLOv11n from Ultralytics.")
    parser.add_argument("--data", type=str, default="../dataset/santascoper.yaml", help="Path to .yaml associated with the training data") 
    parser.add_argument("--epochs", type=int, default=100, help="The number of epochs to train for.")
    parser.add_argument("--patience", type=int, default=15, help="Stop early after n epochs without improvement.")
    parser.add_argument("--batch", type=float, default=16, help="The batch size. Can use a float (i.e. 0.X) to target X percent GPU VRAM utilisation.")
    parser.add_argument("--classes", type=list[int], default=None, help="List of class IDs to train on. Useful for focusing on certain classes during training." )

    args = parser.parse_args()
    return vars(args)


def train_santa_scoper(**args):
    model = YOLO(model=args["model"], task="detect", verbose=False)
    results = model.train(**args)
    return results

if __name__ == "__main__":
    print(f"Fine-Tuning SantaScoper(tm)...\n" + "_" * 50 + "\n")
    kwargs = get_args_as_kwargs()
    results = train_santa_scoper(**kwargs)



