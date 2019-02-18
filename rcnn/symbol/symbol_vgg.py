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

# --------------config---------------
ATTENTION = True
NO_HIDDEN = False
CONV_REDUCED = False
SE_MODULE = False
MLPOOL = True
ONLY_ONE = False


# --------------config---------------


def get_vgg_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")

    conv_feat = {'relu5_3': relu5_3, 'relu4_3': relu4_3, 'relu3_3': relu3_3}

    return conv_feat


fc_rela_w = mx.sym.Variable("fc_rela_weight")
fc_rela_b = mx.sym.Variable("fc_rela_bias")


def relation_module(fc_feat, n):
    """
    get relation feature
    :param fc_feat: top deep feature,shape is(n,d)
    :param n: batch_size(rois number)
    :return: relation-weighted feature
    """
    # (n,d)->(nxn,d)
    # [[ 1.  2.  3.]
    # [ 1.  2.  3.]
    # [ 4.  5.  6.]
    # [ 4.  5.  6.]]
    fc_feat_repeat = mx.symbol.repeat(data=fc_feat, repeats=n, axis=0)
    fc_feat_repeat2 = mx.symbol.reshape(data=fc_feat_repeat, shape=(n, n, -1))
    fc_feat_repeat2 = mx.symbol.transpose(data=fc_feat_repeat2, axes=(1, 0, 2))
    # [[ 1.  2.  3.]
    # [ 4.  5.  6.]
    # [ 1.  2.  3.]
    # [ 4.  5.  6.]]
    fc_feat_repeat3 = mx.symbol.reshape(data=fc_feat_repeat2, shape=(n * n, -1))
    # (nxn,d)+(nxn,d)=(nxn,d)
    # fc_feat_repeat_h = mx.symbol.FullyConnected(data=fc_feat_repeat, num_hidden=512, name="fc_feat_repeat_h")
    # fc_feat_repeat3_h = mx.symbol.FullyConnected(data=fc_feat_repeat3, num_hidden=512, name="fc_feat_repeat3_h")
    rela_fc = mx.symbol.broadcast_add(fc_feat_repeat, fc_feat_repeat3)
    rela_fc = mx.symbol.Activation(data=rela_fc, act_type="tanh")
    # (nxn,d)->(nxn,1)->(n,n,1)
    rela_fc = mx.symbol.FullyConnected(data=rela_fc, weight=fc_rela_w, bias=fc_rela_b, num_hidden=1, name="fc_rela")
    rela_fc = mx.symbol.reshape(data=rela_fc, shape=(n, n, 1))
    rela_fc = mx.symbol.Activation(data=rela_fc, act_type="sigmoid")
    # rela_fc = mx.symbol.softmax(data=rela_fc, axis=1)
    # (n,n,d)x(n,n,1)->(n,n,d)
    rela_fc_at = mx.symbol.broadcast_mul(fc_feat_repeat2, rela_fc)
    # rela_fc_at = mx.symbol.sum(rela_fc_at, axis=1)
    rela_fc_at = mx.symbol.max(rela_fc_at, axis=1)

    return fc_feat + rela_fc_at


def se_module(data, num_filter):
    """
    data:feature map,(n,c,h,w)
    num_filter:channel number
    output: reweighting feature map
    """
    body = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='avg', name='se_pool1')
    body = mx.sym.Convolution(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                              name="se_conv1")
    body = mx.symbol.Activation(data=body, act_type='relu', name='se_relu1')
    body = mx.sym.Convolution(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                              name="se_conv2")
    body = mx.symbol.Activation(data=body, act_type='sigmoid', name="se_sigmoid")
    conv3 = mx.symbol.broadcast_mul(data, body)

    return conv3


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
    fc_u = mx.symbol.FullyConnected(data=feat_map_u, weight=fc_u_w, bias=fc_u_b, num_hidden=512, name="fc_u")
    fc_u_r = mx.symbol.reshape(data=fc_u, shape=(-4, n, -1, -2))
    fc_u_t = mx.symbol.transpose(data=fc_u_r, axes=(1, 0, 2))
    # whc:(kxd),output:(n,k)
    fc_h = mx.symbol.FullyConnected(data=hidden_bg, weight=fc_h_w, num_hidden=512, no_bias=1, name="fc_h")
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
    at_vec_r = mx.symbol.reshape(data=at_vec, shape=(0, 0, 1, 1))
    feat_map_at = mx.symbol.broadcast_mul(data, at_vec_r * 1000)

    return feat_map_at


def get_channel_wise_attention_no_h(data, n):
    """
    data:feature map,(n,c,h,w)
    attention vec:(n,c)
    output:at_feature_map,(n,c,h,w)
    """
    # data_bg = mx.symbol.BlockGrad(data=data)
    data_bg = data
    # (nxc,hxw)
    feat_map_u = mx.symbol.reshape(data=data_bg, shape=(-3, -3))
    # wc:(kx(hxw)),bc:(k,),output:(nxc,k)
    fc_u = mx.symbol.FullyConnected(data=feat_map_u, num_hidden=512, name="fc_u")
    # (nxc,k)
    b = mx.symbol.Activation(data=fc_u, act_type="tanh", name="b")
    # wi:(k,),bi:(1,),output:(nxc,1),(n,c)
    fc_b = mx.symbol.FullyConnected(data=b, num_hidden=1, name="fc_b")
    fc_b_r = mx.symbol.reshape(data=fc_b, shape=(-4, n, -1))
    # (n,c)
    at_vec = mx.symbol.softmax(data=fc_b_r, axis=1, name="at_vec")
    if ONLY_ONE:
        # (1,c)
        at_vec = mx.symbol.mean(data=at_vec, axis=0, keepdims=1)
    at_vec_r = mx.symbol.reshape(data=at_vec, shape=(0, 0, 1, 1))
    feat_map_at = mx.symbol.broadcast_mul(data, at_vec_r * 1000)

    return feat_map_at


fc6_w = mx.sym.Variable("fc6_at_weight")
fc6_b = mx.sym.Variable("fc6_at_bias")
fc7_w = mx.sym.Variable("fc7_weight")
fc7_b = mx.sym.Variable("fc7_bias")


def get_fc_net(data):
    """
    data:feature map,(n,c,h,w)
    output:feature_vec,(n,d)
    """
    flatten = mx.symbol.Flatten(data=data, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, weight=fc6_w, bias=fc6_b, num_hidden=4096, name="fc6_at")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5)
    # group 7
    fc7 = mx.symbol.FullyConnected(data=relu6, weight=fc7_w, bias=fc7_b, num_hidden=4096, name="fc7")
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
        name='roi_pool5_1', data=conv_feat['relu5_3'], rois=rois, pooled_size=(7, 7),
        spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    pool5_2 = mx.symbol.ROIPooling(
        name='roi_pool5_2', data=conv_feat['relu4_3'], rois=rois, pooled_size=(7, 7),
        spatial_scale=2.0 / config.RCNN_FEAT_STRIDE)
    pool5_3 = mx.symbol.ROIPooling(
        name='roi_pool5_3', data=conv_feat['relu3_3'], rois=rois, pooled_size=(7, 7),
        spatial_scale=4.0 / config.RCNN_FEAT_STRIDE)
    # pool5_1_context = mx.symbol.ROIPooling(
    #     name='roi_pool5_1', data=conv_feat['relu5_3'], rois=rois, pooled_size=(7, 7),
    #     spatial_scale=2 * 1.0 / config.RCNN_FEAT_STRIDE)
    # L2 normalization(channel/instance)
    pool5_1 = mx.symbol.L2Normalization(data=pool5_1, mode='instance', name='norm_1')
    pool5_2 = mx.symbol.L2Normalization(data=pool5_2, mode='instance', name='norm_2')
    pool5_3 = mx.symbol.L2Normalization(data=pool5_3, mode='instance', name='norm_3')
    # pool5_1_context = mx.symbol.L2Normalization(data=pool5_1_context, mode='instance', name='norm_4')
    # concat
    pool5_pre = mx.symbol.concat(*[pool5_1, pool5_2, pool5_3], name='pool5_pre')
    # scale
    pool5_pre = pool5_pre * 1000
    # pool5_scale = mx.symbol.Convolution(
    #     data=pool5_pre, kernel=(1, 1), pad=(0, 0), num_filter=1280, num_group=1280, name='pool5_scale')
    if CONV_REDUCED:
        pool5 = mx.symbol.Convolution(
            data=pool5_pre, kernel=(1, 1), pad=(0, 0), num_filter=512, name='pool5_reduced')
    else:
        pool5 = pool5_pre

    return pool5


def get_vgg_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN test with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_vgg_conv(data)
    relu5_3 = conv_feat['relu5_3']

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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

    # Fast R-CNN
    if not MLPOOL:
        pool5 = mx.symbol.ROIPooling(
            name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    else:
        # multi-layer pooling
        pool5 = get_multi_layer_feature(conv_feat, rois)

    # no hidden
    if NO_HIDDEN and ATTENTION:
        pool5 = get_channel_wise_attention_no_h(data=pool5, n=config.TEST.RPN_POST_NMS_TOP_N)
        # drop7 = get_fc_net(data=pool5)

    # attention
    if ATTENTION and not NO_HIDDEN:
        for r in xrange(1, 2):
            fc_hidden = get_fc_net(data=pool5)
            pool5 = get_channel_wise_attention(data=pool5, hidden=fc_hidden, n=config.TEST.RPN_POST_NMS_TOP_N)
        # drop7 = get_fc_net(data=pool5)

    if not ATTENTION and SE_MODULE:
        pool5 = se_module(data=pool5, num_filter=1280)

    if not ATTENTION and CONV_REDUCED:
        # group 6
        flatten = mx.symbol.Flatten(data=pool5, name="flatten")
        fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
        relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
        drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
        # group 7
        fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
        relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
        drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    else:
        drop7 = get_fc_net(data=pool5)

    # relation module
    # drop7_r = relation_module(drop7, config.TEST.RPN_POST_NMS_TOP_N)

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes),
                                 name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes),
                                  name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group


def get_vgg_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN end-to-end with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    conv_feat = get_vgg_conv(data)
    relu5_3 = conv_feat['relu5_3']

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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

    # Fast R-CNN
    if not MLPOOL:
        pool5 = mx.symbol.ROIPooling(
            name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    else:
        # multi-layer pooling
        pool5 = get_multi_layer_feature(conv_feat, rois)

    # no hidden
    if NO_HIDDEN and ATTENTION:
        pool5 = get_channel_wise_attention_no_h(data=pool5, n=config.TRAIN.BATCH_ROIS)
        # drop7 = get_fc_net(data=pool5)
    # attention
    if ATTENTION and not NO_HIDDEN:
        for r in xrange(1, 2):
            fc_hidden = get_fc_net(data=pool5)
            pool5 = get_channel_wise_attention(data=pool5, hidden=fc_hidden, n=config.TRAIN.BATCH_ROIS)
        # drop7 = get_fc_net(data=pool5)

    if not ATTENTION and SE_MODULE:
        pool5 = se_module(data=pool5, num_filter=1280)

    if not ATTENTION and CONV_REDUCED:
        # group 6
        flatten = mx.symbol.Flatten(data=pool5, name="flatten")
        fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
        relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
        drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
        # group 7
        fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
        relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
        drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    else:
        drop7 = get_fc_net(data=pool5)

    # relation module
    # drop7_r = relation_module(drop7, config.TRAIN.BATCH_ROIS)

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
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
