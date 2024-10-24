import torch.nn as nn
import torchvision

from .utils import *

from model.LPD import LPD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Faster_RCNN(LPD.LicensePlateDetection):
    def __init__(self, img_size, roi_size, feature_extractor, n_classes = 1):
        super().__init__("FasterRCNN")
        self.backbone = feature_extractor
        self.img_size = img_size  # (height, width)
        self.roi_size = roi_size  # (height, width)
        self.n_classes = n_classes

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backbone = self.backbone.to(device)

        # Use a dummy input to calculate out_channels, out_h, out_w
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

    def forward(self, images, gt_bboxes, gt_classes):
        total_rpn_loss, feature_map, proposals, \
            positive_anc_ind_sep, GT_class_pos = self.rpn(
                images, gt_bboxes, gt_classes)

        pos_proposals_list = []
        batch_size = images.size(dim=0)
        for idx in range(batch_size):
            proposal_idxs = torch.where(positive_anc_ind_sep == idx)[0]
            proposals_sep = proposals[proposal_idxs].detach().clone()
            pos_proposals_list.append(proposals_sep)

        cls_loss = self.classifier(
            feature_map, pos_proposals_list, GT_class_pos)

        total_loss = cls_loss + total_rpn_loss

        return total_loss

    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7, final_nms_thresh=0.3):
        batch_size = images.size(dim=0)
        proposals_final, conf_scores_final, feature_map = self.rpn.inference(
            images, conf_thresh, nms_thresh)
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

                if cls_boxes.numel() > 0 and cls_scores.numel() > 0:  # Kiểm tra không rỗng
                    keep_idx = ops.nms(cls_boxes, cls_scores, final_nms_thresh)

                    keep_boxes.append(cls_boxes[keep_idx])
                    keep_scores.append(cls_scores[keep_idx])
                    keep_classes.append(torch.full((len(keep_idx),), cls))

            # Chọn bounding box có điểm số cao nhất
            if keep_boxes:
                all_boxes = torch.cat(
                    keep_boxes) if keep_boxes else torch.empty((0, 4))
                all_scores = torch.cat(
                    keep_scores) if keep_scores else torch.empty((0,))
                all_classes = torch.cat(keep_classes) if keep_classes else torch.empty(
                    (0,), dtype=torch.int64)

                if all_scores.numel() > 0:
                    # Lấy chỉ số của bounding box có điểm số cao nhất
                    max_idx = torch.argmax(all_scores)
                    # Chỉ thêm một bounding box
                    final_boxes.append(all_boxes[max_idx].unsqueeze(0))
                    # Chỉ thêm điểm số tương ứng
                    final_scores.append(all_scores[max_idx].unsqueeze(0))
                    # Chỉ thêm lớp tương ứng
                    final_classes.append(all_classes[max_idx].unsqueeze(0))
                else:
                    # Hoặc kích thước phù hợp cho bbox
                    final_boxes.append(torch.empty((0, 4)))
                    final_scores.append(torch.empty((0,)))
                    final_classes.append(torch.empty((0,), dtype=torch.int64))
            else:
                final_boxes.append(torch.empty((0, 4)))
                final_scores.append(torch.empty((0,)))
                final_classes.append(torch.empty((0,), dtype=torch.int64))

        return final_boxes, final_scores, final_classes


class FeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, freeze_backbone=False):
        super(FeatureExtractor, self).__init__()
        self.backbone_name = backbone
        self.backbone = self.load_backbone(backbone, pretrained)
        if freeze_backbone:
            self.freeze_backbone()

    def load_backbone(self, backbone, pretrained):
        if backbone == 'resnet50':
            model = torchvision.models.resnet50(pretrained=pretrained)
            # Remove avgpool and FC layer
            return nn.Sequential(*list(model.children())[:-2])
        elif backbone == 'vgg16':
            model = torchvision.models.vgg16(pretrained=pretrained)
            return model.features  # Use only feature layers
        elif backbone == 'mobilenet_v2':
            model = torchvision.models.mobilenet_v2(pretrained=pretrained)
            return model.features  # Use the feature extractor part
        else:
            raise ValueError(f"Backbone {backbone} is not supported.")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x.to(device))
        return x


class ProposalModule(nn.Module):
    def __init__(self, in_features, hidden_dim=512, n_anchors=9, p_dropout=0.3):
        super().__init__()
        self.n_anchors = n_anchors
        self.conv1 = nn.Conv2d(in_features, hidden_dim,
                               kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p_dropout)
        self.conf_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors * 4, kernel_size=1)

    def forward(self, feature_map, pos_anc_ind=None, neg_anc_ind=None, pos_anc_coords=None):
        # determine mode
        if pos_anc_ind is None or neg_anc_ind is None or pos_anc_coords is None:
            mode = 'eval'
        else:
            mode = 'train'

        out = self.conv1(feature_map)
        out = F.relu(self.dropout(out))

        reg_offsets_pred = self.reg_head(out)  # (B, A*4, hmap, wmap)
        conf_scores_pred = self.conf_head(out)  # (B, A, hmap, wmap)

        if mode == 'train':
            # get conf scores
            conf_scores_pos = conf_scores_pred.flatten()[pos_anc_ind]
            conf_scores_neg = conf_scores_pred.flatten()[neg_anc_ind]
            # get offsets for +ve anchors
            offsets_pos = reg_offsets_pred.contiguous().view(-1,
                                                             4)[pos_anc_ind]
            # generate proposals using offsets
            proposals = generate_proposals(pos_anc_coords, offsets_pos)

            return conf_scores_pos, conf_scores_neg, offsets_pos, proposals

        elif mode == 'eval':
            return conf_scores_pred, reg_offsets_pred


class RegionProposalNetwork(nn.Module):
    def __init__(self, img_size, out_size, out_channels, feature_extractor):
        super().__init__()

        self.img_height, self.img_width = img_size
        self.out_h, self.out_w = out_size

        # downsampling scale factor
        self.width_scale_factor = self.img_width // self.out_w
        self.height_scale_factor = self.img_height // self.out_h

        # scales and ratios for anchor boxes
        self.anc_scales = [2, 4, 6]
        self.anc_ratios = [0.5, 1, 1.5]
        self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios)

        # IoU thresholds for +ve and -ve anchors
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3

        # weights for loss
        self.w_conf = 1
        self.w_reg = 5

        self.feature_extractor = feature_extractor
        self.proposal_module = ProposalModule(
            out_channels, n_anchors=self.n_anc_boxes)

    def forward(self, images, gt_bboxes, gt_classes):
        batch_size = images.size(dim=0)
        images = images.to(device)  # Ensure images are on the correct device
        # Ensure ground truth boxes are on the correct device
        gt_bboxes = gt_bboxes.to(device)
        gt_classes = gt_classes.to(device)
        feature_map = self.feature_extractor(images)

        # generate anchors
        anc_pts_x, anc_pts_y = gen_anc_centers(
            out_size=(self.out_h, self.out_w))
        anc_base = gen_anc_base(
            anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
        anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)

        # get positive and negative anchors amongst other things
        gt_bboxes_proj = project_bboxes(
            gt_bboxes, self.width_scale_factor, self.height_scale_factor, mode='p2a')

        positive_anc_ind, negative_anc_ind, GT_conf_scores, \
            GT_offsets, GT_class_pos, positive_anc_coords, \
            negative_anc_coords, positive_anc_ind_sep = get_req_anchors(
                anc_boxes_all, gt_bboxes_proj, gt_classes)

        # pass through the proposal module
        conf_scores_pos, conf_scores_neg, offsets_pos, proposals = self.proposal_module(feature_map, positive_anc_ind,
                                                                                        negative_anc_ind, positive_anc_coords)

        cls_loss = calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size)
        reg_loss = calc_bbox_reg_loss(GT_offsets, offsets_pos, batch_size)

        total_rpn_loss = self.w_conf * cls_loss + self.w_reg * reg_loss

        return total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos

    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        with torch.no_grad():
            batch_size = images.size(dim=0)
            feature_map = self.feature_extractor(images)

            # generate anchors
            anc_pts_x, anc_pts_y = gen_anc_centers(
                out_size=(self.out_h, self.out_w))
            anc_base = gen_anc_base(
                anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
            anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
            anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)

            # get conf scores and offsets
            conf_scores_pred, offsets_pred = self.proposal_module(feature_map)
            conf_scores_pred = conf_scores_pred.reshape(batch_size, -1)
            offsets_pred = offsets_pred.reshape(batch_size, -1, 4)

            # filter out proposals based on conf threshold and nms threshold for each image
            proposals_final = []
            conf_scores_final = []
            for i in range(batch_size):
                conf_scores = torch.sigmoid(conf_scores_pred[i])
                offsets = offsets_pred[i]
                anc_boxes = anc_boxes_flat[i]
                proposals = generate_proposals(anc_boxes, offsets)
                # filter based on confidence threshold
                conf_idx = torch.where(conf_scores >= conf_thresh)[0]
                conf_scores_pos = conf_scores[conf_idx]
                proposals_pos = proposals[conf_idx]
                # filter based on nms threshold
                nms_idx = ops.nms(proposals_pos, conf_scores_pos, nms_thresh)
                conf_scores_pos = conf_scores_pos[nms_idx]
                proposals_pos = proposals_pos[nms_idx]

                proposals_final.append(proposals_pos)
                conf_scores_final.append(conf_scores_pos)

        return proposals_final, conf_scores_final, feature_map


class ClassificationModule(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, p_dropout=0.3):
        super().__init__()
        self.roi_size = roi_size
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)

        self.cls_head = nn.Linear(hidden_dim, n_classes)

    def forward(self, feature_map, proposals_list, gt_classes=None):

        if gt_classes is None:
            mode = 'eval'
        else:
            mode = 'train'

        roi_out = ops.roi_pool(feature_map, proposals_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)

        roi_out = roi_out.squeeze(-1).squeeze(-1)

        out = self.fc(roi_out)
        out = F.relu(self.dropout(out))

        cls_scores = self.cls_head(out)

        if mode == 'eval':
            return cls_scores
        cls_loss = F.cross_entropy(cls_scores, gt_classes.long())

        return cls_loss
