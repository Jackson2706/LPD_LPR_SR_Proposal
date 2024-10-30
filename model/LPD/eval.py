import numpy as np
import torch
from tqdm import tqdm


def evaluate_model(model, eval_dataloader, iou_threshold=0.5, conf_thresh=0.5, nms_thresh=0.7, final_nms_thresh=0.3):
    """
    Args:
        model: Model Faster R-CNN
        eval_dataloader: DataLoader cho tập evaluation
        iou_threshold: Ngưỡng IoU để xác định positive detection
        conf_thresh: Confidence threshold cho RPN
        nms_thresh: NMS threshold cho RPN
        final_nms_thresh: NMS threshold cuối cùng

    Returns:
        Dictionary chứa các metrics: precision, recall, F1-score, mAP
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # Chuyển sang chế độ evaluation

    total_gt = 0
    total_pred = 0
    total_correct = 0
    all_precisions = []
    all_recalls = []

    # Tính IoU giữa predicted và ground truth boxes
    def calculate_iou(box1, box2):
        # box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / (union + 1e-6)

    with torch.no_grad():
        progress_bar = tqdm(eval_dataloader, desc='Evaluating')
        for batch_data in progress_bar:
            images = batch_data[0]
            gt_bboxes = batch_data[1]
            gt_classes = batch_data[3]  # Bỏ qua keypoints

            # Chuyển data lên device
            images = images.to(device)

            # Inference
            pred_boxes, pred_scores, pred_classes = model(
                images,
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh,
                final_nms_thresh=final_nms_thresh
            )

            # Đánh giá cho từng ảnh trong batch
            for i in range(len(images)):
                gt_boxes_img = gt_bboxes[i].to(device)
                gt_classes_img = gt_classes[i].to(device)
                pred_boxes_img = pred_boxes[i]
                pred_scores_img = pred_scores[i]
                pred_classes_img = pred_classes[i]

                total_gt += len(gt_boxes_img)
                total_pred += len(pred_boxes_img)

                # Matching predictions với ground truths
                matched_gt = set()
                correct_predictions = 0

                if len(pred_boxes_img) > 0:  # Kiểm tra có predictions không
                    for pred_idx, pred_box in enumerate(pred_boxes_img):
                        best_iou = 0
                        best_gt_idx = -1

                        # Tìm GT box có IoU cao nhất với prediction
                        for gt_idx, gt_box in enumerate(gt_boxes_img):
                            if gt_idx in matched_gt:
                                continue

                            iou = calculate_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx

                        # Nếu IoU vượt ngưỡng và class dự đoán đúng
                        if best_iou >= iou_threshold and best_gt_idx != -1:
                            if pred_classes_img[pred_idx] == gt_classes_img[best_gt_idx]:
                                correct_predictions += 1
                                matched_gt.add(best_gt_idx)

                total_correct += correct_predictions

                # Tính precision và recall cho ảnh hiện tại
                precision = correct_predictions / len(pred_boxes_img) if len(pred_boxes_img) > 0 else 0
                recall = correct_predictions / len(gt_boxes_img) if len(gt_boxes_img) > 0 else 0

                all_precisions.append(precision)
                all_recalls.append(recall)

    # Tính các metrics tổng thể
    mean_precision = np.mean(all_precisions) if all_precisions else 0
    mean_recall = np.mean(all_recalls) if all_recalls else 0
    f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-6)

    metrics = {
        'Precision': mean_precision,
        'Recall': mean_recall,
        'F1-Score': f1_score,
        'Total GT': total_gt,
        'Total Predictions': total_pred,
        'Correct Predictions': total_correct
    }

    # In kết quả
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics