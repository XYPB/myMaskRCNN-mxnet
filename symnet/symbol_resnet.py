import mxnet as mx
from . import proposal_target

eps=2e-5
use_global_stats=True
workspace = 1024
RPN_FEAT_STRIDE = [64, 32, 16, 8, 4]
# RCNN_FEAT_STRIDE = [32, 16, 8, 4]


def residual_unit(data, num_filter, stride, dim_match, name):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                                no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                                workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name + '_sc')
    sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
    return sum


def get_resnet_feature(data, units, filter_list):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i)
    conv_C2 = unit

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i)
    conv_C3 = unit

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i)
    conv_C4 = unit

    # res5
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
    conv_C5 = unit

    conv_feat = [conv_C5, conv_C4, conv_C3, conv_C2]
    return conv_feat


def get_resnet_conv_down(conv_feat):
    # C5 to P5
    P5 = mx.symbol.Convolution(data=conv_feat[0], kernel=(1, 1), num_filter=256, name="P5_lateral")

    # P5*2 + C4 = P4
    P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type="nearest", workspace=512, name='P5_upsampling', num_args=1)
    P4_la = mx.symbol.Convolution(data=conv_feat[1], kernel=(1, 1), num_filter=256, name='P4_lateral')
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name='P4_clip')
    P4 = mx.symbol.ElementWiseSum(*[P5_clip, P4_la], name='P4_sum')
    P4 = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name='P4_aggregate')
    
    # P4*2 + C3 = P3
    P4_up = mx.symbol.UpSampling(P4, scale=2, sample_type="nearest", workspace=512, name='P4_upsampling', num_args=1)
    P3_la = mx.symbol.Convolution(data=conv_feat[2], kernel=(1, 1), num_filter=256, name='P3_lateral')
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name='P3_clip')
    P3 = mx.symbol.ElementWiseSum(*[P4_clip, P3_la], name='P3_sum')
    P3 = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name='P3_aggregate')
    
    # P3*2 + C2 = P2
    P3_up = mx.symbol.UpSampling(P3, scale=2, sample_type="nearest", workspace=512, name='P3_upsampling', num_args=1)
    P2_la = mx.symbol.Convolution(data=conv_feat[3], kernel=(1, 1), num_filter=256, name='P2_lateral')
    P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name='P2_clip')
    P2 = mx.symbol.ElementWiseSum(*[P3_clip, P2_la], name='P2_sum')
    P2 = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=256, name='P2_aggregate')
    
    # P5 /2 = P6
    P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6_subsampling')

    conv_fpn_feat = {"stride64":P6, "stride32":P5, "stride16":P4, "stride8":P3, "stride4":P2}

    return conv_fpn_feat, [P6, P5, P4, P3, P2]


def get_resnet_top_feature(data, units, filter_list):
    unit = residual_unit(data=data, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
    bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    return pool1


def get_resnet_train(anchor_scales, anchor_ratios, rpn_feature_stride,
                    rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size, rpn_batch_rois,
                    num_classes, rcnn_feature_stride, rcnn_pooled_size, rcnn_batch_size,
                    rcnn_batch_rois, rcnn_fg_fraction, rcnn_fg_overlap, rcnn_bbox_stds,
                    units, filter_list):
    anchor_scales = (8,)
    num_anchors = 3

    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    
    fpn_labels = []
    fpn_bbox_targets = []
    fpn_bbox_weights = []

    # 训练时需要的参数（train.py label shape name）
    for stride in RPN_FEAT_STRIDE:
        fpn_labels.append(mx.symbol.Variable(name='label_stride%s' %stride))
        fpn_bbox_targets.append(mx.symbol.Variable(name='bbox_target_stride%s' %stride))
        fpn_bbox_weights.append(mx.symbol.Variable(name='bbox_weight_stride%s' %stride))
    
    
    # rpn_label = mx.symbol.Variable(name='label')
    # rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    # rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    conv_feat = get_resnet_feature(data, units=units, filter_list=filter_list)
    conv_fpn_feat, _ = get_resnet_conv_down(conv_feat)

    rpn_conv_weight = mx.symbol.Variable('rpn_conv_3x3_weight')
    rpn_conv_bias = mx.symbol.Variable('rpn_conv_3x3_bias')
    rpn_conv_cls_weight = mx.symbol.Variable('rpn_cls_score_weight')
    rpn_conv_cls_bias = mx.symbol.Variable('rpn_cls_score_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_bbox_pred_weight')
    rpn_conv_bbox_bias = mx.symbol.Variable('rpn_bbox_pred_bias')

    rpn_cls_prob_list = []
    rois_pool_list = []
    rpn_bbox_loss_list = []
    labels_list = []
    bbox_target_list = []
    bbox_weight_list = []
    rpn_cls_score_list = []
    for i, stride in enumerate(RPN_FEAT_STRIDE):
        # print(i, stride)
        # print(fpn_bbox_targets[i])
        # print(conv_fpn_feat['stride%s' % stride])
        # print(conv_fpn_feat['stride%s' % stride].infer_shape(data=(2, 3, 1000, 1000)))
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s'%stride],
                                        kernel=(3, 3), pad=(1, 1),
                                        num_filter=512,
                                        weight=rpn_conv_weight,
                                        bias=rpn_conv_bias)
        rpn_relu = mx.symbol.Activation(data=rpn_conv,
                                        act_type="relu",
                                        name="rpn_relu")
        _, output_shape, _ = conv_fpn_feat['stride%s' % stride].infer_shape(data=(1, 3, 1000, 1000))
        # print(output_shape)
        
        # fpn classification
        # ROI Proposal
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                                kernel=(1, 1), pad=(0, 0),
                                                num_filter=2 * num_anchors,
                                                name="rpn_cls_score_stride%s" % stride,
                                                weight=rpn_conv_cls_weight,
                                                bias=rpn_conv_cls_bias)
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                    shape=(0, 2, -1, 0),
                                                    name="rpn_cls_score_reshape%s" %stride)
        rpn_cls_output = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape,
                                                label=fpn_labels[i],
                                                multi_output=True,
                                                normalization='valid',
                                                use_ignore=True,
                                                ignore_label=-1,
                                                name="rpn_cls_output%s" %stride)
        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                    mode="channel",
                                                    name="rpn_cls_prob_stride%s" % stride)
        rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                shape=(0, 2 * num_anchors, -1, 0),
                                                name='rpn_cls_prob_reshape%s' %stride)
        rpn_cls_prob_list.append(rpn_cls_output)
        rpn_cls_score_list.append(rpn_cls_score)

        # fpn bounding box regression
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                                kernel=(1, 1), pad=(0, 0),
                                                num_filter=4 * num_anchors,
                                                name="rpn_bbox_pred_stride%s" % stride,
                                                weight=rpn_conv_bbox_weight,
                                                bias=rpn_conv_bbox_bias)
        rpn_bbox_loss_ = fpn_bbox_weights[i] * mx.symbol.smooth_l1(name='rpn_bbox_loss_%s'%stride,
                                                                    scalar=3.0,
                                                                    data=(rpn_bbox_pred - fpn_bbox_targets[i]))
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss%s'%stride, data=rpn_bbox_loss_, grad_scale=1 / rpn_batch_rois)
        rpn_bbox_loss_list.append(rpn_bbox_loss)

        # rpn proposal
        rois = mx.symbol.contrib.MultiProposal(cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info,     name='rois%s'%stride,
                                                feature_stride=stride, scales=anchor_scales, ratios=anchor_ratios,
                                                rpn_pre_nms_top_n=rpn_pre_topk, rpn_post_nms_top_n=rpn_post_topk,
                                                threshold=rpn_nms_thresh, rpn_min_size=rpn_min_size)
        
        # rcnn roi proposal target
        # print(stride)
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes, op_type='proposal_target',
                                num_classes=num_classes, batch_images=rcnn_batch_size,
                                batch_rois=rcnn_batch_rois, fg_fraction=rcnn_fg_fraction,
                                fg_overlap=rcnn_fg_overlap, box_stds=rcnn_bbox_stds)
        rois = group[0]
        labels_list.append(group[1])
        bbox_target_list.append(group[2])
        bbox_weight_list.append(group[3])

        # rcnn roi pooling
        roi_pool = mx.symbol.contrib.ROIAlign(
                    name='roi_pool%s'%stride, data=conv_fpn_feat['stride%s'%stride], rois=rois,
                    pooled_size=(14, 14),
                    spatial_scale=1.0 / stride)
        rois_pool_list.append(roi_pool)


    # rpn网络的rois
    rois_align_concat = mx.symbol.Concat(*rois_pool_list, dim=0)
    # rpn 网络的输出
    rpn_cls_score_concat = mx.symbol.Concat(*rpn_cls_score_list, dim=0, name='rpn_cls_score_concat')
    rpn_cls_output_concat = mx.symbol.Concat(*rpn_cls_prob_list, dim=0, name='rpn_cls_output_concat')
    rpn_bbox_loss_concat = mx.symbol.Concat(*rpn_bbox_loss_list, dim=0, name='rpn_bbox_loss_concat')
    labels_list_concat = mx.symbol.Concat(*labels_list, dim=0, name='labels_list_concat')
    # rpn网络loss的weight和target
    bbox_target_concat = mx.symbol.Concat(*bbox_target_list, dim=0, name='bbox_target_concat')
    bbox_weight_concat = mx.symbol.Concat(*bbox_weight_list, dim=0, name='bbox_weight_concat')

    # rcnn top feature
    # top_feat = get_resnet_top_feature(roi_pool, units=units, filter_list=filter_list)
    # 删掉res5，直接把roi结果喂到rcnn cla，bbox，mask；
    # Mask

    # rcnn classification

    flatten = mx.symbol.Flatten(data=rois_align_concat, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024, name='fc6')
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=1024, name='fc7')
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_relu7")

    cls_score = mx.symbol.FullyConnected(name='cls_score', data=relu7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_list_concat, normalization='batch')

    # rcnn bbox regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=relu7, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight_concat * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target_concat))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / rcnn_batch_rois)

    # reshape output
    label = mx.symbol.Reshape(data=labels_list_concat, shape=(rcnn_batch_size, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(rcnn_batch_size, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(rcnn_batch_size, -1, 4 * num_classes), name='bbox_loss_reshape')

    # group output
    # print(mx.symbol.BlockGrad(label))

    group = mx.symbol.Group([rpn_cls_prob_list[0], rpn_cls_prob_list[1], rpn_cls_prob_list[2], rpn_cls_prob_list[3], rpn_cls_prob_list[4],
                            rpn_bbox_loss_list[0], rpn_bbox_loss_list[1], rpn_bbox_loss_list[2], rpn_bbox_loss_list[3], rpn_bbox_loss_list[4],
                            cls_prob, bbox_loss,
                            mx.symbol.BlockGrad(label)])
    # group = mx.symbol.Group([rpn_cls_output_concat,
    #                         rpn_bbox_loss_concat,
    #                         cls_prob, bbox_loss,
    #                         mx.symbol.BlockGrad(label)])
    return group


def get_resnet_test(anchor_scales, anchor_ratios, rpn_feature_stride,
                    rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size,
                    num_classes, rcnn_feature_stride, rcnn_pooled_size, rcnn_batch_size,
                    units, filter_list):

    anchor_scales = (8,)
    num_anchors = 3

    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_resnet_feature(data, units=units, filter_list=filter_list)
    conv_fpn_feat, _ = get_resnet_conv_down(conv_feat)

    rpn_conv_weight = mx.symbol.Variable('rpn_conv_3x3_weight')
    rpn_conv_bias = mx.symbol.Variable('rpn_conv_3x3_bias')
    rpn_conv_cls_weight = mx.symbol.Variable('rpn_cls_score_weight')
    rpn_conv_cls_bias = mx.symbol.Variable('rpn_cls_score_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_bbox_pred_weight')
    rpn_conv_bbox_bias = mx.symbol.Variable('rpn_bbox_pred_bias')

    rois_pool_list = []
    rois_list = []
    for i, stride in enumerate(RPN_FEAT_STRIDE):
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s'%stride],
                                        kernel=(3, 3), pad=(1, 1),
                                        num_filter=512,
                                        weight=rpn_conv_weight,
                                        bias=rpn_conv_bias)
        rpn_relu = mx.symbol.Activation(data=rpn_conv,
                                        act_type="relu",
                                        name="rpn_relu")
        _, output_shape, _ = conv_fpn_feat['stride%s' % stride].infer_shape(data=(1, 3, 1000, 1000))
        # print(output_shape)
        
        # fpn classification
        # ROI Proposal
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                                kernel=(1, 1), pad=(0, 0),
                                                num_filter=2 * num_anchors,
                                                name="rpn_cls_score_stride%s" % stride,
                                                weight=rpn_conv_cls_weight,
                                                bias=rpn_conv_cls_bias)
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                    shape=(0, 2, -1, 0),
                                                    name="rpn_cls_score_reshape%s" %stride)
        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                    mode="channel",
                                                    name="rpn_cls_prob_stride%s" % stride)
        rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                shape=(0, 2 * num_anchors, -1, 0),
                                                name='rpn_cls_prob_reshape%s' %stride)

        # fpn bounding box regression
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                                kernel=(1, 1), pad=(0, 0),
                                                num_filter=4 * num_anchors,
                                                name="rpn_bbox_pred_stride%s" % stride,
                                                weight=rpn_conv_bbox_weight,
                                                bias=rpn_conv_bbox_bias)

        # rpn proposal
        rois = mx.symbol.contrib.MultiProposal(cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois%s'%stride,
                                                feature_stride=stride, scales=anchor_scales, ratios=anchor_ratios,
                                                rpn_pre_nms_top_n=rpn_pre_topk, rpn_post_nms_top_n=rpn_post_topk,
                                                threshold=rpn_nms_thresh, rpn_min_size=rpn_min_size)

        # rcnn roi pooling
        rois_list.append(rois)
        roi_pool = mx.symbol.contrib.ROIAlign(
                    name='roi_pool%s'%stride, data=conv_fpn_feat['stride%s'%stride], rois=rois,
                    pooled_size=(14, 14),
                    spatial_scale=1.0 / stride)
        rois_pool_list.append(roi_pool)

    # rpn网络的rois
    rois_align_concat = mx.symbol.Concat(*rois_pool_list, dim=0)
    rois_concat = mx.symbol.Concat(*rois_list, dim=0)
    # rpn 网络的输出
    # rpn网络loss的weight和target

    # rcnn top feature
    # top_feat = get_resnet_top_feature(roi_pool, units=units, filter_list=filter_list)
    # 删掉res5，直接把roi结果喂到rcnn cla，bbox，mask；
    # Mask

    # rcnn classification

    flatten = mx.symbol.Flatten(data=rois_align_concat, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024, name='fc6')
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=1024, name='fc7')
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_relu7")

    cls_score = mx.symbol.FullyConnected(name='cls_score', data=relu7, num_hidden=num_classes)
    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)

    # rcnn bbox regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=relu7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(rcnn_batch_size, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(rcnn_batch_size, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    # print(mx.symbol.BlockGrad(label))

    group = mx.symbol.Group([rois_concat, cls_prob, bbox_pred])
    return group
