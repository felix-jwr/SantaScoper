from ultralytics import YOLO
from get_eta import get_eta

if __name__ == "__main__":
    print(f"Testing SantaScoper(tm)...\n" + "=" * 50 + "\n")
    model = YOLO(model="./model/SantaScoper/weights/best.pt", 
                 task="detect", 
                 verbose=False)
    preds = model("dataset/visdrone/images/test/9999986_00000_d_0000012.jpg")
    # preds = model("dataset/santascoper/images/val/0000356_02941_d_0000644.jpg")
    # preds = model("dataset/windowsxpgrassland_wsanta.jpg")

    for idx, pred in enumerate(preds):
        pth = f"out/pred_{idx}.jpg"
        pred.save(filename=pth)
        get_eta(img_path=pth)
