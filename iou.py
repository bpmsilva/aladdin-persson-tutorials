"""
Aladdin Persson Intersection over Union YouTube video
"""
import torch

def intersect_over_union(pred_boxes, label_boxes, box_format="corners"):
    """
    Calculates intersection over union
    Parameters:
        pred_boxes (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        label_boxes (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    # pred_boxes is (N, 4), where N is the number of bounding boxes
    # label_boxes is (N, 4), where N is the number of bounding boxes
    if box_format == 'corners':
        # box1
        box1_x1 = pred_boxes[..., 0:1]
        box1_y1 = pred_boxes[..., 1:2]
        box1_x2 = pred_boxes[..., 2:3]
        box1_y2 = pred_boxes[..., 3:4]

        # box2
        box2_x1 = label_boxes[..., 0:1]
        box2_y1 = label_boxes[..., 1:2]
        box2_x2 = label_boxes[..., 2:3]
        box2_y2 = label_boxes[..., 3:4]

    elif box_format == 'midpoint':
        # box1
        width1 = pred_boxes[..., 2:3] / 2
        heigh1 = pred_boxes[..., 3:4] / 2
        box1_x1 = pred_boxes[..., 0:1] - width1
        box1_y1 = pred_boxes[..., 1:2] - heigh1
        box1_x2 = pred_boxes[..., 0:1] + width1
        box1_y2 = pred_boxes[..., 1:2] + heigh1

        # box2
        width2 = label_boxes[..., 2:3] / 2
        height2 = label_boxes[..., 3:4] / 2
        box2_x1 = label_boxes[..., 0:1] - width2
        box2_y1 = label_boxes[..., 1:2] - height2
        box2_x2 = label_boxes[..., 0:1] + width2
        box2_y2 = label_boxes[..., 1:2] + height2

    else:
        raise Exception('Unknown bounding box format.')

    # intersection
    x_1 = torch.max(box1_x1, box2_x1)
    y_1 = torch.max(box1_y1, box2_y1)
    x_2 = torch.min(box1_x2, box2_x2)
    y_2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x_2 - x_1).clamp(0) * (y_2 - y_1).clamp(0)

    # union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)
