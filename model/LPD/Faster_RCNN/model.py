from .RCNN_Head import ClassificationModule
from .RPN import RegionProposalNetwork
from .utils import *

from model.LPD import LPD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Faster_RCNN(LPD.LicensePlateDetection):
    def __init__(self, img_size, roi_size, feature_extractor, n_classes=1):
        super().__init__("FasterRCNN")
        self.backbone = feature_extractor
        self.img_size = img_size  # (height, width)
        self.roi_size = roi_size  # (height, width)
        self.n_classes = n_classes

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backbone = self.backbone.to(device)

        dummy_input = torch.randn(1, 3, self.img_size[0], self.img_size[1]).to(device)
        out = self.backbone(dummy_input)
        out_c, out_h, out_w = out.size(1), out.size(2), out.size(3)

        self.out_size = (out_h, out_w)
        self.out_channels = out_c

        # Calculate scale factors
        self.width_scale_factor = self.img_size[1] // out_w
        self.height_scale_factor = self.img_size[0] // out_h

        self.rpn = RegionProposalNetwork(
            self.img_size, self.out_size, self.out_channels, feature_extractor)
        self.classifier = ClassificationModule(
            self.out_channels, n_classes, roi_size)

    def forward(self, images, gt_bboxes=None, gt_classes=None, conf_thresh=0.5, nms_thresh=0.7, final_nms_thresh=0.3):
        # Training mode
        if self.training and gt_bboxes is not None and gt_classes is not None:
            total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos = self.rpn(
                images, gt_bboxes, gt_classes)

            pos_proposals_list = []
            batch_size = images.size(dim=0)
            for idx in range(batch_size):
                proposal_idxs = torch.where(positive_anc_ind_sep == idx)[0]
                proposals_sep = proposals[proposal_idxs].detach().clone()
                pos_proposals_list.append(proposals_sep)

            cls_loss = self.classifier(feature_map, pos_proposals_list, GT_class_pos)
            total_loss = cls_loss + total_rpn_loss
            return total_loss

        # Inference mode
        else:
            batch_size = images.size(dim=0)
            proposals_final, conf_scores_final, feature_map = self.rpn.inference(
                images, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
            cls_scores = self.classifier(feature_map, proposals_final)

            cls_probs = F.softmax(cls_scores, dim=-1)
            classes_all = torch.argmax(cls_probs, dim=-1)

            final_boxes = []
            final_scores = []
            final_classes = []

            c = 0
            for i in range(batch_size):
                n_proposals = len(proposals_final[i])
                boxes = proposals_final[i]
                scores = conf_scores_final[i]
                classes = classes_all[c: c + n_proposals]
                c += n_proposals

                # Áp dụng NMS cho mỗi lớp riêng biệt
                keep_boxes = []
                keep_scores = []
                keep_classes = []

                for cls in torch.unique(classes):
                    cls_mask = (classes == cls)
                    cls_boxes = boxes[cls_mask]
                    cls_scores = scores[cls_mask]

                    if cls_boxes.numel() > 0 and cls_scores.numel() > 0:
                        keep_idx = ops.nms(cls_boxes, cls_scores, final_nms_thresh)

                        keep_boxes.append(cls_boxes[keep_idx])
                        keep_scores.append(cls_scores[keep_idx])
                        keep_classes.append(torch.full((len(keep_idx),), cls))

                # Chọn bounding box có điểm số cao nhất
                if keep_boxes:
                    all_boxes = torch.cat(keep_boxes) if keep_boxes else torch.empty((0, 4))
                    all_scores = torch.cat(keep_scores) if keep_scores else torch.empty((0,))
                    all_classes = torch.cat(keep_classes) if keep_classes else torch.empty((0,), dtype=torch.int64)

                    if all_scores.numel() > 0:
                        max_idx = torch.argmax(all_scores)
                        final_boxes.append(all_boxes[max_idx].unsqueeze(0))
                        final_scores.append(all_scores[max_idx].unsqueeze(0))
                        final_classes.append(all_classes[max_idx].unsqueeze(0))
                    else:
                        final_boxes.append(torch.empty((0, 4)))
                        final_scores.append(torch.empty((0,)))
                        final_classes.append(torch.empty((0,), dtype=torch.int64))
                else:
                    final_boxes.append(torch.empty((0, 4)))
                    final_scores.append(torch.empty((0,)))
                    final_classes.append(torch.empty((0,), dtype=torch.int64))

            return final_boxes, final_scores, final_classes






