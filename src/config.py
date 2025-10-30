VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

VOC_IMG_MEAN = (0.45286129, 0.43170348, 0.39989259)  # RGB
VOC_IMG_STD = (0.2770844, 0.27359877, 0.2856848)  # RGB
COLORS = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

# network expects a square input of this dimension
# YOLO v3 standard is 416x416 (divisible by 32)
YOLO_IMG_DIM = 416
ANCHORS = [
    [(0.8070, 0.4611), (0.5324, 0.7788), (0.8743, 0.8733)],
    [(0.1728, 0.4558), (0.4204, 0.4361), (0.2917, 0.6999)],
    [(0.0558, 0.0889), (0.1188, 0.2081), (0.3135, 0.2341)],
]

GRID_SIZES = [13, 26, 52]