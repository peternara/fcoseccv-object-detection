import torch
from torch.utils.data import Dataset, DataLoader
from eccv_vid import EccvDataset
import cv2

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

CATEGORIES = [
        "__background__ ",
        "person",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "bus",
        "motor",
    ]
def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors
def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 2
        )

    return image
def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    # scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    scores = [1 for _ in range(len(labels))]
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image

eccv = EccvDataset('/media/e813/E/dataset/eccv/eccv/VisDrone2018-VID-train')
for img,box,idx in eccv:
    img = overlay_class_names(img,box)
    img = overlay_boxes(img,box)
    print(img.shape)
    cv2.imshow('sd',img)
    cv2.waitKey(100)


