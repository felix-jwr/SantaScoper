from ultralytics import YOLO

if __name__ == "__main__":
    print(f"Testing SantaScoper(tm)...\n" + "=" * 50 + "\n")
    model = YOLO(model="./model/SantaScoper/weights/best.pt", 
                 task="detect", 
                 verbose=False)
    preds = model("dataset/visdrone/images/test/9999986_00000_d_0000012.jpg")
    # preds = model("dataset/santascoper/images/val/0000356_02941_d_0000644.jpg")
    # preds = model("dataset/windowsxpgrassland_wsanta.jpg")

    for idx, pred in enumerate(preds):
        pred.save(filename=f"pred_{idx}.jpg")
