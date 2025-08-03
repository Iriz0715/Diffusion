import numpy as np
import pandas as pd
import tensorflow as tf


# TODO: Put this function to a common script for both deployment and development.
def get_strides_list(layer_number, im_size, mode='symmetric'):
    # mode: symmetric. e.g. [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    # mode: last. e.g. [[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
    s_list = np.zeros([layer_number, len(im_size)], dtype=np.uint8)
    for i in range(layer_number):
        # stop pooling when size is odd && size <= 4
        to_pool = (np.array(im_size) % 2 ** (i + 1) == 0).astype(int)
        to_pool *= ((np.array(im_size) // 2 ** (i + 1)) > 4).astype(int)
        s_list[i] = 1 + to_pool
    if mode == 'symmetric':
        strides_list = np.concatenate([s_list[::-2], s_list[layer_number % 2::2]])
    else:
        strides_list = np.array(s_list)
    return strides_list

def build_inference_model(model, config):
    im_size = list(model.input_shape[1:-1])
    inputs = tf.keras.layers.Input(shape=model.input_shape[1:])
    predictions = model(inputs)
    strides_list = get_strides_list(config['layers'], im_size, mode='last')  # stride 1 last
    anchor_strides = np.stack([np.prod(strides_list[:i+1], axis=0) for i in range(len(strides_list))], axis=0)
    anchor_strides = anchor_strides[config['start_level'] : config['start_level'] + config['num_head_levels']]
    anchor_scales = config['anchor_scales']
    anchor_box = AnchorBox(aspect_ratios=config['aspect_ratios'], scales=anchor_scales, strides=anchor_strides)
    prediction_decoder = DecodePredictions(anchor_box=anchor_box,
                                           num_classes=config['nclass'],
                                           confidence_threshold=config['confidence_threshold'],
                                           nms_iou_threshold=config['nms_iou_threshold'],
                                           max_detections_per_class=config['max_detections_per_class'],
                                           max_detections=config['max_detections'], 
                                           box_variance=config['box_variance'])
    outputs = prediction_decoder(inputs, predictions)
    inference_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return inference_model

def extract_bboxes_from_annotation(annotation, return_categories=False, return_scores=False):
    bboxes = []
    categories = []
    scores = []
    for anno in annotation:
        if 'bboxes' not in annotation[anno]:
            continue
        bbox_dicts = annotation[anno]['bboxes']
        for bbox in bbox_dicts:
            bboxes.append([bbox['z'], bbox['y'], bbox['x'], 
                           bbox['zRange'], bbox['yRange'], bbox['xRange']])
            if return_categories:
                categories.append(bbox.get('category_id', 0))
            if return_scores:
                scores.append(bbox.get('confidence', 0))
    
    if len(bboxes) == 0:
        bboxes = [[]]  # 2D array
    
    out = [bboxes]
    if return_categories:
        out.append(categories)
    if return_scores:
        out.append(scores)
    
    if len(out) == 1:
        return out[0]
    return tuple(out)

# default iou
def compute_iou(boxA, boxB, epsilon=1e-8, GIoU=False):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes_pred: A tensor with shape `(N, 6)` representing bounding boxes
        where each box is of the format `[x, y, z, x_range, y_range, z_range]`.
        boxes_gt: A tensor with shape `(M, 6)` representing bounding boxes
        where each box is of the format `[x, y, z, x_range, y_range, z_range]`.

        No matter x, y, z are the center points or corner points.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxA = tf.cast(boxA, tf.float32)
    boxB = tf.cast(boxB, tf.float32)
    if len(boxA.shape) == 1:
        boxA = tf.expand_dims(boxA, axis=0)
    if len(boxB.shape) == 1:
        boxB = tf.expand_dims(boxB, axis=0)
    
    rA = boxA[..., 3:]
    rB = boxB[..., 3:]
    inf_boxA = boxA[..., :3] - rA / 2
    inf_boxB = boxB[..., :3] - rB / 2
    sup_boxA = boxA[..., :3] + rA / 2
    sup_boxB = boxB[..., :3] + rB / 2
    sup = tf.minimum(sup_boxA[:, None, :], sup_boxB)
    inf = tf.maximum(inf_boxA[:, None, :], inf_boxB)
    intersect = tf.maximum(0.0, sup - inf)
    interArea = intersect[:,:,0] * intersect[:,:,1] * intersect[:,:,2]
    boxAArea = boxA[:, 3] * boxA[:, 4] * boxA[:, 5]
    boxBArea = boxB[:, 3] * boxB[:, 4] * boxB[:, 5]

    unionArea = tf.maximum(boxAArea[:, None] + boxBArea - interArea, epsilon)
    
    iou = interArea / unionArea
    
    if GIoU:
        gsup = tf.maximum((boxA[:, :3] + boxA[:, 3:] / 2)[:, None, :], boxB[:, :3] + boxB[:, 3:] / 2)
        ginf = tf.minimum((boxA[:, :3] - boxA[:, 3:] / 2)[:, None, :], boxB[:, :3] - boxB[:, 3:] / 2)
        gvol = tf.maximum(gsup - ginf, 0)
        gvolArea = (gvol[:, :, 0] * gvol[:, :, 1] * gvol[:, :, 2])
        iou = interArea / unionArea - (gvolArea - unionArea) / gvolArea
    
    return iou

def compute_distance_hit(gt_bboxes, pred_bboxes, spacings=[1.0, 1.0, 1.0], return_info=False):
    gt_bboxes = np.array(gt_bboxes).reshape([-1, 6])
    pred_bboxes = np.array(pred_bboxes).reshape([-1, 6])
    gt_bboxes = gt_bboxes * np.append(spacings, spacings)
    pred_bboxes = pred_bboxes * np.append(spacings, spacings)
    
    gt_center_radius = np.zeros([len(gt_bboxes), 4])
    pred_center_radius = np.zeros([len(pred_bboxes), 4])
    gt_bboxes = np.array(gt_bboxes)
    gt_center_radius[:, :3] = gt_bboxes[:, :3] + gt_bboxes[:, 3:] / 2
    gt_center_radius[:, 3] = np.sqrt(np.mean(np.square(gt_bboxes[:, 3:]), axis=1)) / 2
    pred_bboxes = np.array(pred_bboxes)
    pred_center_radius[:, :3] = pred_bboxes[:, :3] + pred_bboxes[:, 3:] / 2
    pred_center_radius[:, 3] = np.sqrt(np.mean(np.square(pred_bboxes[:, 3:]), axis=1))
    distance = np.sqrt(np.mean(np.square(gt_center_radius[:, None, :3] - pred_center_radius[:, :3]), axis=-1))
    hit = (distance < gt_center_radius[:, None, 3]).astype(np.float32)
    if return_info:
        return hit, distance, gt_center_radius, pred_center_radius
    return hit

def compute_hit_tpr_fpr(gt_bboxes, pred_bboxes, spacings=[1.0, 1.0, 1.0], metric='distance', iou_threshold=0.4):
    """
    voxel_spacing: should be corresponding to bboxes coordinates
    iou_threshold: only valid when metric=='iou'
    """
    gt_bboxes = np.array(gt_bboxes).reshape([-1, 6])
    pred_bboxes = np.array(pred_bboxes).reshape([-1, 6])
    if gt_bboxes.shape[0] == 0:
        return 0.0, 1.0
    if pred_bboxes.shape[0] == 0:
        return 0.0, 0.0
    gt_bboxes = gt_bboxes * np.append(spacings, spacings)
    pred_bboxes = pred_bboxes * np.append(spacings, spacings)
    if metric == 'iou':
        iou_matrix = compute_iou(gt_bboxes, pred_bboxes)
        hit_matrix = iou_matrix > iou_threshold
    elif metric == 'distance':
        hit_matrix = compute_distance_hit(gt_bboxes, pred_bboxes)
    hit_by_gt = np.amax(hit_matrix, axis=1)
    tpr = np.mean(hit_by_gt)
    hit_by_pred = np.amax(hit_matrix, axis=0)
    fpr = np.mean(hit_by_pred == 0)
    return tpr, fpr

# # distance (safely overwrite the iou function)
# def compute_iou_distance(boxA, boxB):
#     """Computes pairwise IOU matrix for given two sets of boxes

#     Arguments:
#       boxes_pred: A tensor with shape `(N, 4)` representing bounding boxes
#         where each box is of the format `[x, y, z, diameter]`.
#         boxes_gt: A tensor with shape `(M, 4)` representing bounding boxes
#         where each box is of the format `[x, y, z, diameter]`.

#     Returns:
#       pairwise IOU matrix with shape `(N, M)`, where the value at ith row
#         jth column holds the IOU between ith box and jth box from
#         boxes1 and boxes2 respectively.
#     """
#     boxA = tf.cast(boxA, tf.float32)  # pred boxes
#     boxB = tf.cast(boxB, tf.float32)  # gt boxes
#     if len(boxA.shape) == 1:
#         boxA = tf.expand_dims(boxA, axis=0)
#     if len(boxB.shape) == 1:
#         boxB = tf.expand_dims(boxB, axis=0)
    
#     rA = boxA[..., 3:]/2.0
#     rB = boxB[..., 3:]/2.0
#     diff = tf.tile(boxA[:, None, :3], [1, len(boxB), 1]) - tf.tile(boxB[None, :, :3], [len(boxA), 1, 1])
#     dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))
#     iou = (1.0 - (dist - rB[None, :, 0]) / rB[None, :, 0]) / 2.0
    
#     return tf.clip_by_value(iou, 0.0, 1.0)

### Anchor: Z, Y, X, Diameter
class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[4, 8, 16, 32]`. Where each anchor each box is of the
    format `[x, y, z, x_range, y_range, z_range]`. where x, y, z is the center point.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the base (smallest) scale of the anchor boxes
        at each location on the first feature map. 
        [[x_scale_1, y_scale_1, z_scale_1], [x_scale_2, y_scale_2, z_scale_2], ...] (n_scales, n_dim)
        each scale is the real anchor size on original resolution map
        e.g. [[4, 6, 6], [5, 8, 8], [6, 10, 10]] 3 anchor scales on strides=4 feature map.
        The anchor scales on other deeper feature maps will be doubled by strides.
        e.g. [[8, 12, 12], [10, 16, 16], [12, 20, 20]] on strides=8 feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self, aspect_ratios=[0.5, 1.0, 2.0],
                 scales=[[4, 4, 4], [5, 6, 6], [6, 8, 8]],
                 strides=[[4, 4, 4], [8, 8, 8], [16, 16, 16], [16, 32, 32]]):
        self.aspect_ratios = aspect_ratios
        self.scales = np.asarray(scales, np.float32)
        self.strides = np.asarray(strides, np.float32)

        self._num_anchors = len(self.aspect_ratios) * len(self.aspect_ratios) * len(self.scales) #n_xy * n_xz * n_scales
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        # aspect ratio just for y and z (x is always the axial axis in CT images)
        ratios = tf.sqrt(self.aspect_ratios)
        n_ar = len(self.aspect_ratios)
        y_anchors = tf.reshape(tf.round(ratios[:, None] * tf.constant(self.scales[:, 1], tf.float32)[None, :], 0), [-1])
        z_anchors = tf.reshape(tf.round((1 / ratios)[:, None] * tf.constant(self.scales[:, 2], tf.float32)[None, :], 0), [-1])
        
        anchor_dims = []
        for dim_y, dim_z in zip(y_anchors, z_anchors):
            for dim_x in self.scales[:, 0]:
                anchor_dims.append([dim_x, dim_y, dim_z])
        anchor_dims = tf.stack(anchor_dims)
        anchor_dims_all_levels = [anchor_dims]
        for l in range(1, len(self.strides)):
            anchor_dims_all_levels.append(anchor_dims * tf.divide(self.strides[l], self.strides[0]))
        anchor_dims_all_levels = tf.stack(anchor_dims_all_levels, axis=0) # (n_levels, n_anchors, 3)
        return anchor_dims_all_levels

    def _get_anchors(self, x_size, y_size, z_size, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(x_range * y_range * z_range * num_anchors, 6)`
        """
        ry = tf.range(y_size, dtype=tf.float32) + 0.5
        rx = tf.range(x_size, dtype=tf.float32) + 0.5
        rz = tf.range(z_size, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry, rz, indexing='ij'), axis=-1) * self.strides[level]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level][None, None, None, :, :], [x_size, y_size, z_size, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [x_size * y_size * z_size * self._num_anchors, 6]
        )

    def get_anchors(self, im_size=[80, 192, 192]):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 6)`
        """
        feature_map_shapes = [tf.divide(tf.cast(im_size, tf.float32), self.strides[i]) 
                              for i in range(len(self.strides))]
        
        anchors = [
            self._get_anchors(
                tf.cast(feature_map_shapes[i][0], tf.int32),
                tf.cast(feature_map_shapes[i][1], tf.int32),
                tf.cast(feature_map_shapes[i][2], tf.int32),
                i,
            )
            for i in range(len(feature_map_shapes))
        ]
        return tf.concat(anchors, axis=0)

class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self, anchor_box, box_variance=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2], match_iou=0.5, ignore_iou=0.4):
        self._anchor_box = anchor_box
        self._box_variance = tf.convert_to_tensor(
            box_variance,
            dtype=tf.float32
        )
        self.match_iou = match_iou
        self.ignore_iou = ignore_iou
        print('encode:', self._box_variance)

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 6)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, z, x_range, y_range, z_range]`.
          gt_boxes: A float tensor with shape `(num_objects, 6)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, z, x_range, y_range, z_range]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to be ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        low_quality_match = tf.maximum(tf.reduce_max(max_iou), 0.1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        positive_mask = tf.where(tf.equal(max_iou, low_quality_match), True, positive_mask)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))

#         tf.print(tf.shape(matched_gt_idx),
#              tf.reduce_sum(tf.cast(tf.greater(matched_gt_idx, tf.zeros_like(matched_gt_idx)), dtype=tf.float32)),
#              tf.shape(anchor_boxes), tf.shape(gt_boxes),
#              matched_gt_idx,
#              tf.reduce_sum(tf.cast(positive_mask, dtype=tf.float32)), 
#              tf.reduce_sum(tf.cast(negative_mask, dtype=tf.float32)), 
#              tf.reduce_sum(tf.cast(ignore_mask, dtype=tf.float32)),
#              low_quality_match)

        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [   # [(x, y, z)_gt - (x, y, z)_a] / (rx, ry, rz)
                (matched_gt_boxes[:, :3] - anchor_boxes[:, :3]) / anchor_boxes[:, 3:],
                # log[(rx, ry, rz)_gt / (rx, ry, rz)_a]
                tf.math.log(matched_gt_boxes[:, 3:] / anchor_boxes[:, 3:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape)
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes, self.match_iou, self.ignore_iou
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids  # -1: negative; 0+: positive with ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)  # -2: ignore
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([cls_target, box_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape[1:4], gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        return batch_images, labels.stack()

class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the FPN model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        anchor_box,
        num_classes=1,
        confidence_threshold=0.05,
        nms_iou_threshold=0.35,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = anchor_box
        self._box_variance = tf.convert_to_tensor(
            box_variance, dtype=tf.float32
        )
        # print('decode:', self._box_variance)
        # print('nclass:', self.num_classes)
        
    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes_transformed = tf.concat(
            [
                boxes[:, :, :3] * anchor_boxes[:, :, 3:] + anchor_boxes[:, :, :3],
                tf.math.exp(boxes[:, :, 3:]) * anchor_boxes[:, :, 3:],
            ],
            axis=-1,
        )
        return boxes_transformed

    def _nms_eager(self, bboxes, scores):
        if bboxes.shape[0] == 0:
            return bboxes, scores
        if len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
            scores = np.expand_dims(scores, axis=0)
            return bboxes, scores 
        pick = np.array([], dtype=np.int32)
        idxs = np.argsort(scores)
        while len(idxs) > 0 and len(pick) < self.max_detections_per_class:
            last = idxs[-1]
            pick = np.append(pick, last)
            if len(idxs) == 1:
                break
            others = np.take(bboxes, idxs[:-1], axis=0)
            if len(others.shape) == 1:
                others = np.expand_dims(others, axis=0)
            last_bbox = bboxes[last]
            if len(last_bbox.shape) == 1:
                last_bbox = np.expand_dims(last_bbox, axis=0)
            iou_array = compute_iou(others, last_bbox)
            not_overlap = np.squeeze(iou_array < self.nms_iou_threshold)
            idxs = idxs[:-1][not_overlap]

        # in case the pick=[]
        if len(pick) == 0:
            return np.zeros([0, 1]), np.zeros([0])
        return np.take(bboxes, pick, axis=0), np.take(scores, pick, axis=0)

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1:4])
        all_box_predictions = predictions[:, :, self.num_classes:]
        all_cls_predictions = tf.nn.sigmoid(predictions[:, :, :self.num_classes])[0]  # get the 0th batch
        all_boxes = self._decode_box_predictions(anchor_boxes[None, ...], all_box_predictions)[0]
        
        all_detected_boxes = tf.zeros([0, 6])
        all_confidence_level = tf.zeros([0,])
        all_detected_labels = tf.zeros([0,])
        for c in range(self.num_classes):
            # filter by confidence threshold
            # only for class c nms
            pass_threshold = all_cls_predictions[..., c] > self.confidence_threshold
            pass_threshold = tf.logical_and(pass_threshold, tf.reduce_all(all_boxes > 0, axis=-1))
            pass_threshold = tf.squeeze(tf.where(pass_threshold)) # to gather index
            cls_predictions = tf.gather(all_cls_predictions[..., c], pass_threshold, axis=0)  # (len(boxes), classes)
            boxes = tf.gather(all_boxes, pass_threshold, axis=0)
            detected_boxes, confidence_level = tf.py_function(self._nms_eager, 
                                                              [boxes, cls_predictions], 
                                                              [tf.float32, tf.float32])
            detected_labels = tf.ones_like(confidence_level) * c
            
            all_detected_boxes = tf.concat([all_detected_boxes, detected_boxes], axis=0)
            all_confidence_level = tf.concat([all_confidence_level, confidence_level], axis=0)
            all_detected_labels = tf.concat([all_detected_labels, detected_labels], axis=0)

        selected_idxs = tf.argsort(all_confidence_level, direction='DESCENDING')[:self.max_detections]
        return (tf.gather(all_detected_boxes, selected_idxs), 
                tf.gather(all_confidence_level, selected_idxs), 
                tf.gather(all_detected_labels, selected_idxs))

def check_if_true_or_false_positive(annotations, detections, iou_threshold):
    annotations = np.array(annotations, dtype=np.float64)
    scores = []
    false_positives = []
    true_positives = []
    detected_annotations = [] # a GT box should be mapped only one predicted box at most.
    for d in detections:
        scores.append(d[0])
        if len(annotations) == 0:
            false_positives.append(1)
            true_positives.append(0)
            continue
        overlaps = compute_iou(annotations, d[1:])
        assigned_annotation = np.argmax(overlaps)
        max_overlap = overlaps[assigned_annotation]
        if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
            false_positives.append(0)
            true_positives.append(1)
            detected_annotations.append(assigned_annotation)
        else:
            false_positives.append(1)
            true_positives.append(0)
    return scores, false_positives, true_positives

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_real_annotations(table):
    res = dict()
    ids = table['ImageID'].values.astype(str)
    labels = table['LabelName'].values.astype(str)
    xctr = table['XCtr'].values.astype(np.float32)
    x_range = table['XRange'].values.astype(np.float32)
    yctr = table['YCtr'].values.astype(np.float32)
    y_range = table['YRange'].values.astype(np.float32)
    zctr = table['ZCtr'].values.astype(np.float32)
    z_range = table['ZRange'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xctr[i], yctr[i], zctr[i], x_range[i], y_range[i], z_range[i]]
        res[id][label].append(box)

    return res

def get_detections(table):
    res = dict()
    ids = table['ImageID'].values.astype(str)
    labels = table['LabelName'].values.astype(str)
    scores = table['Conf'].values.astype(np.float32)
    xctr = table['XCtr'].values.astype(np.float32)
    x_range = table['XRange'].values.astype(np.float32)
    yctr = table['YCtr'].values.astype(np.float32)
    y_range = table['YRange'].values.astype(np.float32)
    zctr = table['ZCtr'].values.astype(np.float32)
    z_range = table['ZRange'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [scores[i], xctr[i], yctr[i], zctr[i], x_range[i], y_range[i], z_range[i]]
        res[id][label].append(box)
    return res

def mean_average_precision_for_boxes(ann, pred, iou_threshold=0.4, exclude_not_in_annotations=False, verbose=True):
    """
    :param ann: path to CSV-file with annotations or numpy array of shape (N, 8)
    :param pred: path to CSV-file with predictions (detections) or numpy array of shape (N, 9)
    :param iou_threshold: IoU between boxes which count as 'match'. Default: 0.5
    :param exclude_not_in_annotations: exclude image IDs which are not exist in annotations. Default: False
    :param verbose: print detailed run info. Default: True
    :return: tuple, where first value is mAP and second values is dict with AP for each class.
    """

    valid = pd.DataFrame(ann, columns=['ImageID', 'LabelName', 'XCtr', 'YCtr', 'ZCtr', 'XRange', 'YRange', 'ZRange'])
    preds = pd.DataFrame(pred, columns=['ImageID', 'LabelName', 'Conf', 'XCtr', 'YCtr', 'ZCtr', 'XRange', 'YRange', 'ZRange'])
    ann_unique = valid['ImageID'].unique()
    preds_unique = preds['ImageID'].unique()

    if verbose:
        print('Number of files in annotations: {}'.format(len(ann_unique)))
        print('Number of files in predictions: {}'.format(len(preds_unique)))

    # Exclude files not in annotations!
    if exclude_not_in_annotations:
        preds = preds[preds['ImageID'].isin(ann_unique)]
        preds_unique = preds['ImageID'].unique()
        if verbose:
            print('Number of files in detection after reduction: {}'.format(len(preds_unique)))

    unique_classes = valid['LabelName'].unique().astype(str)
    if verbose:
        print('Unique classes: {}'.format(len(unique_classes)))

    all_detections = get_detections(preds)
    all_annotations = get_real_annotations(valid)
    if verbose:
        print('Detections length: {}'.format(len(all_detections)))
        print('Annotations length: {}'.format(len(all_annotations)))

    average_precisions = {}
    for zz, label in enumerate(sorted(unique_classes)):

        # Negative class
        if str(label) == 'nan':
            continue

        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0.0

        for i in range(len(ann_unique)):
            detections = []
            annotations = []
            id = ann_unique[i]
            if id in all_detections:
                if label in all_detections[id]:
                    detections = all_detections[id][label]
            if id in all_annotations:
                if label in all_annotations[id]:
                    annotations = all_annotations[id][label]

            if len(detections) == 0 and len(annotations) == 0:
                continue
                
            num_annotations += len(annotations)
            
            scr, fp, tp = check_if_true_or_false_positive(annotations, detections, iou_threshold)
            scores += scr
            false_positives += fp
            true_positives += tp

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        false_positives = np.array(false_positives)
        true_positives = np.array(true_positives)
        scores = np.array(scores)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations, precision, recall
        if verbose:
            s1 = "{:30s} | {:.6f} | {:7d}".format(label, average_precision, int(num_annotations))
            print(s1)

    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations, _, _) in average_precisions.items():
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision
    mean_ap = precision / present_classes
    if verbose:
        print('mAP: {:.6f}'.format(mean_ap))
    return mean_ap, average_precisions