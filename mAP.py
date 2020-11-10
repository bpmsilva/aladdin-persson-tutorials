import torch
from collections import Counter
from iou import intersect_over_union

# 7. Do all of that for all IoU thresholds
def average_precision(
    # all prediction boxes of considered data set
    pred_boxes, # pred_boxes (list): [[img_idx, class_pred, prob_score, x1, y1, x2, y2], [...], ...]
    true_boxes,
    iou_threshold=0.5,
    box_format='corners',
    num_classes=20
):
    average_precisions = []
    epsilon = 1e-6

    # 6. Do all for all classes
    for c in range(num_classes):
        # 1. Get all bounding boxes (predictions and groun-truths)
        detections = [detection for detection in pred_boxes if detection[1] == c]
        ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]

        # 2. Sort by descending confidence score
        detections.sort(key=lambda x: x[2], reverse=True)

        # 3. Calculate the precision and recall as we go through all the outputs
        # dictionary containing the number of bboxes for each image
        amount_bboxes = Counter([gt[0] for gt in ground_truths]) # amount_bboxes = {0: 3, 1: 5, ...}

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val) # amount_bboxes = {0: torch.tensor([0, 0, 0]), 1: torch.tensor([0, 0, 0, 0, 0]), ...}

        # initialize some variables
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            # we only compare bboxes and detections of the same images
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            
            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                iou = intersect_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        
        # 4. "Plot" the Precision-Recall graph
        # [1, 1, 0, 1, 0] - > [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.device(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))

        # 5. Calculate the Area under the PR curve
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precision)
