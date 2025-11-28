# SantaDetector 🎅
Simple dockerised demo for training and testing YOLOv11 models on VisDrone2019, with additional scripts for manual augmentation of images (randomly add Santa into VisDrone images).

Uses (mostly) default Ultralytics configurations for ease.
**Note:** Doesn't include datasets or trained model weights.

## Quickstart

1. Download the VisDrone2019 dataset from the [official website](http://aiskyeye.com/) or [GitHub](https://github.com/VisDrone/VisDrone-Dataset).

2. `docker build -t santascoper:latest`

3. `sudo docker run -it --rm --ipc=host --gpus all -v "$PWD":/opt/app -w /opt/app santascoper`

4. `python ./src/train.py --optional args --go here`

## Santa Augmentation

To augment images to add Santa, use `insert_santas.py`, and specify directories for the dataset images and labels, and a directory containing Santa images (`.png` recommended). You'll also need to provide a probability with which to perform the augmentation.

You also need to manually specify the `class_id` for Santa. Defaults to 10 which is the appropriate value VisDrone, otherwise use the your dataset's highest `class_id`+1.
 
## Results

Training runs default to the `models/` directory, with a default run subfolder of `SantaScoper/`:

```
model/
└── SantaScoper/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    │   [ ... ]
    └── results.png
```

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv11 implementation
- [VisDrone](http://aiskyeye.com/) for the dataset
- Licensed under the GNU Affero General Public License v3.0
