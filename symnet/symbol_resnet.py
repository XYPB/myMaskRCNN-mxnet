import mxnet as mx
from . import proposal_target, assign_rois, proposal_fpn

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

    # e.g. For ResNet50, the feature is :
    # outputs = ['stage1_activation2', 'stage2_activation3',
    #            'stage3_activation5', 'stage4_activation2']
    # with regard to [conv2, conv3, conv4, conv5] -> [C2, C3, C4, C5]
    # append more layers with reversed order : [P5, P4, P3, P2]
    y = conv_feat[0]
    base_features = conv_feat
    num_filters = [256, 256, 256, 256]
    num_stages = len(num_filters) + 1  # usually 5
    weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2.)
    tmp_outputs = []
    # num_filter is 256 in ori paper
    for i, (bf, f) in enumerate(zip(base_features, num_filters)):
        if i == 0:
            y = mx.sym.Convolution(y, num_filter=f, kernel=(1, 1), pad=(0, 0),
                                    stride=(1, 1), no_bias=False,
                                    name="P{}_conv_lat".format(num_stages - i),
                                    attr={'__init__': weight_init})
            
            y_p6 = mx.sym.Convolution(y, num_filter=f, kernel=(3, 3), pad=(1, 1),
                                        stride=(2, 2), no_bias=False,
                                        name='P{}_conv1'.format(num_stages + 1),
                                        attr={'__init__': weight_init})
        else:
            bf = mx.sym.Convolution(bf, num_filter=f, kernel=(1, 1), pad=(0, 0),
                                    stride=(1, 1), no_bias=False,
                                    name="P{}_conv_lat".format(num_stages - i),
                                    attr={'__init__': weight_init})
            y = mx.sym.UpSampling(y, scale=2, sample_type='nearest',
                                    name="P{}_upsp".format(num_stages - i))

            # make two symbol alignment
            # method 1 : mx.sym.Crop
            # y = mx.sym.Crop(*[y, bf], name="P{}_clip".format(num_stages-i))
            # method 2 : mx.sym.slice_like
            y = mx.sym.slice_like(y, bf * 0, axes=(2, 3),
                                    name="P{}_clip".format(num_stages - i))
            y = mx.sym.ElementWiseSum(bf, y, name="P{}_sum".format(num_stages - i))
        # Reduce the aliasing effect of upsampling described in ori paper
        out = mx.sym.Convolution(y, num_filter=f, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                                    no_bias=False, name='P{}_conv1'.format(num_stages - i),
                                    attr={'__init__': weight_init})
        
        tmp_outputs.append(out)
    P2, P3, P4, P5 = tuple(tmp_outputs[::-1])
    P6 = y_p6

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride64":P6, "stride32":P5, "stride16":P4, "stride8":P3, "stride4":P2})

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
    # for stride in RPN_FEAT_STRIDE:
    #     l_tmp = mx.symbol.Variable(name='label_stride%s' %stride)
    #     l_tmp_reshape = mx.symbol.Reshape(data=l_tmp,
    #                                               shape=(0, 0, -1),
    #                                               name="l_tmp_reshape_stride%s" % stride)
    #     fpn_labels.append(l_tmp_reshape)
    #     rpn_bbox_gt = mx.symbol.Variable(name='bbox_target_stride%s' %stride)
    #     rpn_bbox_gt_reshape = mx.symbol.Reshape(data=rpn_bbox_gt,
    #                                               shape=(0, 0, -1),
    #                                               name="rpn_bbox_gt_reshape_stride%s" % stride)
    #     fpn_bbox_targets.append(rpn_bbox_gt_reshape)
    #     rpn_bbox_gt_w = mx.symbol.Variable(name='bbox_weight_stride%s' %stride)
    #     rpn_bbox_gt_w_reshape = mx.symbol.Reshape(data=rpn_bbox_gt_w,
    #                                               shape=(0, 0, -1),
    #                                               name="rpn_bbox_weights_reshape_stride%s" % stride)
    #     fpn_bbox_weights.append(rpn_bbox_gt_w_reshape)
    fpn_labels = mx.symbol.Variable(name='label')
    fpn_bbox_targets = mx.symbol.Variable(name='bbox_target')
    fpn_bbox_weights = mx.symbol.Variable(name='bbox_weight')
    
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

    rois_pool_list = []
    rois_list = []
    rpn_bbox_pred_list = []
    rpn_bbox_loss_list = []
    labels_list = []
    bbox_target_list = []
    bbox_weight_list = []
    rpn_cls_score_list = []
    rpn_cls_prob_dict = {}
    rpn_bbox_pred_dict = {}
    for i, stride in enumerate(RPN_FEAT_STRIDE):
        # print(i, stride)
        # print(fpn_bbox_targets[i])
        # print(conv_fpn_feat['stride%s' % stride])
        # print(conv_fpn_feat['stride%s' % stride].infer_shape(data=(2, 3, 1000, 1000)))
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s'%stride],
                                        kernel=(3, 3), pad=(1, 1),
                                        num_filter=1024,
                                        weight=rpn_conv_weight,
                                        bias=rpn_conv_bias)
        rpn_relu = mx.symbol.Activation(data=rpn_conv,
                                        act_type="relu",
                                        name="rpn_relu")
        # _, output_shape, _ = conv_fpn_feat['stride%s' % stride].infer_shape(data=(1, 3, 1000, 1000))
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
                                                    shape=(0, 2, -1),
                                                    name="rpn_cls_score_reshape%s" %stride)
        rpn_cls_score_list.append(rpn_cls_score_reshape)
        rpn_cls_score_reshape_1 = mx.symbol.Reshape(data=rpn_cls_score,
                                                    shape=(0, 2, -1, 0),
                                                    name="rpn_cls_score_reshape_1_%s" %stride)
        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape_1,
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
        
        rpn_bbox_pred_reshape = mx.symbol.Reshape(data=rpn_bbox_pred,
                                                  shape=(0, 0, -1),
                                                  name="rpn_bbox_pred_reshape_stride%s" % stride)
        rpn_cls_prob_dict.update({'cls_prob_stride%s'%stride:rpn_cls_prob_reshape})
        rpn_bbox_pred_dict.update({'bbox_pred_stride%s'%stride:rpn_bbox_pred})

    
    rpn_cls_score_list = mx.symbol.concat(*rpn_cls_score_list, dim=2, name="rpn_cls_score_list_concat")

    rpn_cls_output = mx.symbol.SoftmaxOutput(data=rpn_cls_score_list,
                                            label=fpn_labels,
                                            multi_output=True,
                                            normalization='valid',
                                            use_ignore=True,
                                            ignore_label=-1,
                                            name="rpn_cls_output")
    
    rpn_bbox_pred = mx.symbol.concat(*rpn_bbox_pred_list, dim=2, name="rpn_bbox_pred_list_concat")

    rpn_bbox_loss_ = fpn_bbox_weights * mx.symbol.smooth_l1(name='rpn_bbox_loss_%s'%stride,
                                                                scalar=3.0,
                                                                data=(rpn_bbox_pred - fpn_bbox_targets))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss%s'%stride, data=rpn_bbox_loss_, grad_scale=1 / rpn_batch_rois)


    # rpn proposal
    args_dict = {}
    args_dict.update(rpn_cls_prob_dict)
    args_dict.update(rpn_bbox_pred_dict)
    aux_dict = {'im_info':im_info,'name':'rois',
                'op_type':'proposal_fpn','output_score':False,
                'feat_stride':RPN_FEAT_STRIDE,'scales':tuple(anchor_scales),
                'ratios':tuple(anchor_ratios),
                'rpn_pre_nms_top_n':rpn_pre_topk,
                'rpn_post_nms_top_n':rpn_post_topk,
                'threshold':rpn_nms_thresh}
    args_dict.update(aux_dict)
    # Proposal
    rois_concat = mx.symbol.Custom(**args_dict)
    group = mx.symbol.Custom(rois=rois_concat, gt_boxes=gt_boxes, op_type='proposal_target',
                            num_classes=num_classes, batch_images=rcnn_batch_size,
                            batch_rois=rcnn_batch_rois, fg_fraction=rcnn_fg_fraction,
                            fg_overlap=rcnn_fg_overlap, box_stds=rcnn_bbox_stds)
    
    rois_concat = group[0]
    labels_list_concat = group[1]
    bbox_target_concat = group[2]
    bbox_weight_concat = group[3]


    # rpn网络的rois
    # rpn 网络的输出
    
    _, x1, y1, x2, y2 = mx.symbol.split(rois_concat, axis=-1, num_outputs=5)
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    roi_level = mx.symbol.floor(4 + mx.symbol.log2(mx.symbol.sqrt(w * h) / 224.0 + eps))
    roi_level = mx.symbol.squeeze(mx.symbol.clip(roi_level, 2, 5))
    # [2,2,..,3,3,...,4,4,...,5,5,...] ``Prohibit swap order here``
    # roi_level_sorted_args = mx.symbol.argsort(roi_level, is_ascend=True)
    # roi_level = mx.symbol.sort(roi_level, is_ascend=True)
    # rpn_rois = mx.symbol.take(rpn_rois, roi_level_sorted_args, axis=0)
    pooled_roi_feats = []
    for i, stride in enumerate(RPN_FEAT_STRIDE[1:]):
        # Pool features with all rois first, and then set invalid pooled features to zero,
        # at last ele-wise add together to aggregate all features.
        pooled_feature = mx.symbol.contrib.ROIAlign(conv_fpn_feat['stride%s' % stride], rois_concat, (7, 7),
                                            1. / stride,
                                            sample_ratio=2)
        pooled_feature = mx.symbol.where(roi_level == 5 - i, pooled_feature, mx.symbol.zeros_like(pooled_feature))
        pooled_roi_feats.append(pooled_feature)
    # Ele-wise add to aggregate all pooled features
    rois_align_concat = mx.symbol.ElementWiseSum(*pooled_roi_feats)

    # stride = 32
    # roi_pool = mx.symbol.contrib.ROIAlign(
    #         name='roi_pool%s'%stride, data=conv_fpn_feat['stride%s'%stride], rois=rois_concat,
    #         pooled_size=(7, 7),
    #         spatial_scale=1.0 / stride)
    # rois_align_concat = roi_pool
    # rcnn top feature
    # top_feat = get_resnet_top_feature(roi_pool, units=units, filter_list=filter_list)
    # 删掉res5，直接把roi结果喂到rcnn cla，bbox，mask；
    # Mask

    # rcnn classification

    flatten = mx.symbol.Flatten(data=rois_align_concat, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024, name='fc6')
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.symbol.FullyConnected(data=relu6, num_hidden=1024, name='fc7')
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

    group = mx.symbol.Group([rpn_cls_output,
                            rpn_bbox_loss,
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

    rpn_cls_prob_dict = {}
    rpn_bbox_pred_dict = {}
    for i, stride in enumerate(RPN_FEAT_STRIDE):
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s'%stride],
                                        kernel=(3, 3), pad=(1, 1),
                                        num_filter=1024,
                                        weight=rpn_conv_weight,
                                        bias=rpn_conv_bias)
        rpn_relu = mx.symbol.Activation(data=rpn_conv,
                                        act_type="relu",
                                        name="rpn_relu")
        # _, output_shape, _ = conv_fpn_feat['stride%s' % stride].infer_shape(data=(1, 3, 1000, 1000))
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
        rpn_cls_prob_dict.update({'cls_prob_stride%s'%stride:rpn_cls_prob_reshape})
        rpn_bbox_pred_dict.update({'bbox_pred_stride%s'%stride:rpn_bbox_pred})

        # rpn proposal

    args_dict = {}
    args_dict.update(rpn_cls_prob_dict)
    args_dict.update(rpn_bbox_pred_dict)
    aux_dict = {'im_info':im_info,'name':'rois',
                'op_type':'proposal_fpn','output_score':False,
                'feat_stride':RPN_FEAT_STRIDE,'scales':tuple(anchor_scales),
                'ratios':tuple(anchor_ratios),
                'rpn_pre_nms_top_n':rpn_pre_topk,
                'rpn_post_nms_top_n':rpn_post_topk,
                'threshold':rpn_nms_thresh}
    args_dict.update(aux_dict)
    # Proposal
    rois_concat = mx.symbol.Custom(**args_dict)
    
    
    _, x1, y1, x2, y2 = mx.symbol.split(rois_concat, axis=-1, num_outputs=5)
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    roi_level = mx.symbol.floor(4 + mx.symbol.log2(mx.symbol.sqrt(w * h) / 224.0 + eps))
    roi_level = mx.symbol.squeeze(mx.symbol.clip(roi_level, 2, 5))
    # [2,2,..,3,3,...,4,4,...,5,5,...] ``Prohibit swap order here``
    # roi_level_sorted_args = mx.symbol.argsort(roi_level, is_ascend=True)
    # roi_level = mx.symbol.sort(roi_level, is_ascend=True)
    # rpn_rois = mx.symbol.take(rpn_rois, roi_level_sorted_args, axis=0)
    pooled_roi_feats = []
    for i, stride in enumerate(RPN_FEAT_STRIDE[1:]):
        # Pool features with all rois first, and then set invalid pooled features to zero,
        # at last ele-wise add together to aggregate all features.
        pooled_feature = mx.symbol.contrib.ROIAlign(conv_fpn_feat['stride%s' % stride], rois_concat, (7, 7),
                                            1. / stride,
                                            sample_ratio=2)
        pooled_feature = mx.symbol.where(roi_level == 5 - i, pooled_feature, mx.symbol.zeros_like(pooled_feature))
        pooled_roi_feats.append(pooled_feature)
    # Ele-wise add to aggregate all pooled features
    rois_align_concat = mx.symbol.ElementWiseSum(*pooled_roi_feats)
    

    # group = mx.symbol.Custom(
    #     rois=rois_concat,
    #     P2=conv_fpn_feat['stride4'],
    #     P3=conv_fpn_feat['stride8'],
    #     P4=conv_fpn_feat['stride16'],
    #     P5=conv_fpn_feat['stride32'],
    #     op_type='assign_rois'
    #     )
    # rois_align_concat = group[0]
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
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.symbol.FullyConnected(data=relu6, num_hidden=1024, name='fc7')
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
