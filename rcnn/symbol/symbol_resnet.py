# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
from rcnn.config import config
from . import proposal
from . import proposal_target

eps = 2e-5
use_global_stats = True
workspace = 512
res_deps = {'50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
# units = res_deps['101']
units = res_deps['50']
filter_list = [256, 512, 1024, 2048]


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


def get_resnet_conv(data):
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
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True,
                             name='stage1_unit%s' % i)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True,
                             name='stage2_unit%s' % i)
    stride8 = unit

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True,
                             name='stage3_unit%s' % i)
    stride16 = unit

    # res5
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                             name='stage4_unit%s' % i)
    stride32 = unit

    conv_feats = {'stride32': stride32, 'stride16': stride16, 'stride8': stride8}
    return conv_feats


fc6_w = mx.sym.Variable("fc6_weight")
fc6_b = mx.sym.Variable("fc6_bias")
fc7_w = mx.sym.Variable("fc7_weight")
fc7_b = mx.sym.Variable("fc7_bias")


def get_fc_net(data):
    flatten = mx.symbol.Flatten(data=data, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, weight=fc6_w, bias=fc6_b, num_hidden=1024, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5)
    # group 7
    fc7 = mx.symbol.FullyConnected(data=relu6, weight=fc7_w, bias=fc7_b, num_hidden=1024, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    # drop7 = mx.symbol.Dropout(data=relu7, p=0.5)

    return relu7


def get_multi_layer_feature(conv_feat, rois):
    """
    multi-layer pooling
    :param conv_feat: multi-layer feature
    :param rois: proposals
    :return: pooled multi-layer fusion feature
    """
    pool5_1 = mx.symbol.ROIPooling(
        name='roi_pool5_1', data=conv_feat['stride32'], rois=rois, pooled_size=(7, 7),
        spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    pool5_2 = mx.symbol.ROIPooling(
        name='roi_pool5_2', data=conv_feat['stride16'], rois=rois, pooled_size=(7, 7),
        spatial_scale=2.0 / config.RCNN_FEAT_STRIDE)
    pool5_3 = mx.symbol.ROIPooling(
        name='roi_pool5_3', data=conv_feat['stride8'], rois=rois, pooled_size=(7, 7),
        spatial_scale=4.0 / config.RCNN_FEAT_STRIDE)
    # L2 normalization
    pool5_1 = mx.symbol.L2Normalization(data=pool5_1, mode='instance', name='norm_1')
    pool5_2 = mx.symbol.L2Normalization(data=pool5_2, mode='instance', name='norm_2')
    pool5_3 = mx.symbol.L2Normalization(data=pool5_3, mode='instance', name='norm_3')
    # concat
    pool5_pre = mx.symbol.concat(*[pool5_1, pool5_2, pool5_3], name='pool5_pre')
    # scale
    # 1,10,100 all ok
    pool5_pre = pool5_pre * 1000
    # pool5_scale = mx.symbol.Convolution(
    #     data=pool5_pre, kernel=(1, 1), pad=(0, 0), num_filter=1280, num_group=1280, name='pool5_scale')
    # pool5 = mx.symbol.Convolution(
    #     data=pool5_pre, kernel=(1, 1), pad=(0, 0), num_filter=1024, name='pool5_reduced')
    pool5 = pool5_pre

    return pool5


fc_u_w = mx.sym.Variable("fc_u_weight")
fc_u_b = mx.sym.Variable("fc_u_bias")
fc_h_w = mx.sym.Variable("fc_h_weight")
fc_b_w = mx.sym.Variable("fc_b_weight")
fc_b_b = mx.sym.Variable("fc_b_bias")


def get_channel_wise_attention(data, hidden, n):
    """
    data:feature map,(n,c,h,w)
    hidden:fc,(n,hidden)
    n:batch size
    attention vec:(n,c)
    output:at_feature_map,(n,c,h,w)
    """
    # data_bg = mx.symbol.BlockGrad(data=data)
    # hidden_bg = mx.symbol.BlockGrad(data=hidden)
    data_bg = data
    hidden_bg = hidden
    # (nxc,hxw)
    feat_map_u = mx.symbol.reshape(data=data_bg, shape=(-3, -3))
    # wc:(kx(hxw)),bc:(k,),output:(nxc,k),(n,c,k),(c,n,k)
    fc_u = mx.symbol.FullyConnected(data=feat_map_u, weight=fc_u_w, bias=fc_u_b, num_hidden=1024, name="fc_u")
    fc_u_r = mx.symbol.reshape(data=fc_u, shape=(-4, n, -1, -2))
    fc_u_t = mx.symbol.transpose(data=fc_u_r, axes=(1, 0, 2))
    # whc:(kxd),output:(n,k)
    fc_h = mx.symbol.FullyConnected(data=hidden_bg, weight=fc_h_w, num_hidden=1024, no_bias=1, name="fc_h")
    # (c,n,k)
    fc_add = mx.symbol.broadcast_add(fc_u_t, fc_h)
    b = mx.symbol.Activation(data=fc_add, act_type="tanh", name="b")
    # (cxn,k)
    b_r = mx.symbol.reshape(data=b, shape=(-3, -2))
    # wi:(k,),bi:(1,),output:(cxn,1),(c,n),(n,c)
    fc_b = mx.symbol.FullyConnected(data=b_r, weight=fc_b_w, bias=fc_b_b, num_hidden=1, name="fc_b")
    fc_b_r = mx.symbol.reshape(data=fc_b, shape=(-4, -1, n))
    fc_b_t = mx.symbol.transpose(data=fc_b_r, axes=(1, 0))
    # (n,c)
    at_vec = mx.symbol.softmax(data=fc_b_t, axis=1, name="at_vec")
    # at_vec = mx.symbol.Activation(data=fc_b_t, act_type="sigmoid", name="at_vec")
    at_vec_r = mx.symbol.reshape(data=at_vec, shape=(0, 0, 1, 1))
    feat_map_at = mx.symbol.broadcast_mul(data, at_vec_r * 1000)
    # feat_map_at = mx.symbol.broadcast_mul(data, at_vec_r)

    return feat_map_at


def get_resnet_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    conv_feats = get_resnet_conv(data)
    conv_feat = conv_feats['stride32']

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                           data=(rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                    grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    if config.TRAIN.CXX_PROPOSAL:
        rois = mx.symbol.contrib.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES),
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                             num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                             batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # # Fast R-CNN
    # roi_pool = mx.symbol.ROIPooling(
    #     name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # # multi-layer pooling
    roi_pool = get_multi_layer_feature(conv_feats, rois)
    # attention
    for r in xrange(1, 2):
        fc_hidden = get_fc_net(roi_pool)
        roi_pool = get_channel_wise_attention(data=roi_pool, hidden=fc_hidden, n=config.TRAIN.BATCH_ROIS)
    pool1 = get_fc_net(roi_pool)

    # # res5
    # unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    # for i in range(2, units[3] + 1):
    #     unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
    #                          name='stage4_unit%s' % i)
    # bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
    # relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool1, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=pool1, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes),
                                 name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes),
                                  name='bbox_loss_reshape')

    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    return group


def get_resnet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feats = get_resnet_conv(data)
    conv_feat = conv_feats['stride32']

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    if config.TEST.CXX_PROPOSAL:
        rois = mx.symbol.contrib.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES),
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # # Fast R-CNN
    # roi_pool = mx.symbol.ROIPooling(
    #     name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # # multi-layer pooling
    roi_pool = get_multi_layer_feature(conv_feats, rois)
    # attention
    for r in xrange(1, 2):
        fc_hidden = get_fc_net(roi_pool)
        roi_pool = get_channel_wise_attention(data=roi_pool, hidden=fc_hidden, n=config.TEST.RPN_POST_NMS_TOP_N)
    pool1 = get_fc_net(roi_pool)

    # # res5
    # unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    # for i in range(2, units[3] + 1):
    #     unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
    #                          name='stage4_unit%s' % i)
    # bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
    # relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool1, num_hidden=num_classes)
    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=pool1, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes),
                                 name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes),
                                  name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group
