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
    data_bg = mx.symbol.BlockGrad(data=data)
    hidden_bg = mx.symbol.BlockGrad(data=hidden)
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


fc6_w = mx.sym.Variable("fc6_at_weight")
fc6_b = mx.sym.Variable("fc6_at_bias")
fc7_w = mx.sym.Variable("fc7_weight")
fc7_b = mx.sym.Variable("fc7_bias")


def get_fc_net(data):
    flatten = mx.symbol.Flatten(data=data, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, weight=fc6_w, bias=fc6_b, num_hidden=4096, name="fc6_at")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6_%s"%name)
    # group 7
    fc7 = mx.symbol.FullyConnected(data=relu6, weight=fc7_w, bias=fc7_b, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    # drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7_%s"%name)

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
    # L2 normalization(channel/instance)
    pool5_1 = mx.symbol.L2Normalization(data=pool5_1, mode='instance', name='norm_1')
    pool5_2 = mx.symbol.L2Normalization(data=pool5_2, mode='instance', name='norm_2')
    pool5_3 = mx.symbol.L2Normalization(data=pool5_3, mode='instance', name='norm_3')
    # concat
    pool5_pre = mx.symbol.concat(*[pool5_1, pool5_2, pool5_3], name='pool5_pre')
    # scale
    pool5_pre = pool5_pre * 1000
    # pool5_scale = mx.symbol.Convolution(
    #     data=pool5_pre, kernel=(1, 1), pad=(0, 0), num_filter=1280, num_group=1280, name='pool5_scale')
    # pool5 = mx.symbol.Convolution(
    #     data=pool5_pre, kernel=(1, 1), pad=(0, 0), num_filter=512, name='pool5_reduced')
    pool5 = pool5_pre

    return pool5


def get_adaptive_pooling_feature(conv_feat, rois):
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

    # pool5_1 = mx.symbol.Convolution(
    #     data=pool5_1, kernel=(1, 1), pad=(0, 0), num_filter=512, name='pool5_1_conv')
    # pool5_2 = mx.symbol.Convolution(
    #     data=pool5_2, kernel=(1, 1), pad=(0, 0), num_filter=512, name='pool5_2_conv')
    # pool5_3 = mx.symbol.Convolution(
    #     data=pool5_3, kernel=(1, 1), pad=(0, 0), num_filter=512, name='pool5_3_conv')

    flatten_1 = mx.symbol.Flatten(data=pool5_1, name="flatten_1")
    fc6_1 = mx.symbol.FullyConnected(data=flatten_1, num_hidden=4096, name="fc6_1")
    relu6_1 = mx.symbol.Activation(data=fc6_1, act_type="relu", name="relu6_1")
    flatten_2 = mx.symbol.Flatten(data=pool5_2, name="flatten_2")
    fc6_2 = mx.symbol.FullyConnected(data=flatten_2, num_hidden=4096, name="fc6_2")
    relu6_2 = mx.symbol.Activation(data=fc6_2, act_type="relu", name="relu6_2")
    flatten_3 = mx.symbol.Flatten(data=pool5_3, name="flatten_3")
    fc6_3 = mx.symbol.FullyConnected(data=flatten_3, num_hidden=4096, name="fc6_3")
    relu6_3 = mx.symbol.Activation(data=fc6_3, act_type="relu", name="relu6_3")

    pool5 = relu6_1 + relu6_2 + relu6_3

    # pool5_scale = mx.symbol.Convolution(
    #     data=pool5_pre, kernel=(1, 1), pad=(0, 0), num_filter=1280, num_group=1280, name='pool5_scale')
    # pool5 = mx.symbol.Convolution(
    #     data=pool5_pre, kernel=(1, 1), pad=(0, 0), num_filter=512, name='pool5_reduced')

    return pool5


def get_vgg_conv_dowm(conv_feat):
    # C5 to P5, 1x1 dimension reduction to 256
    P5 = mx.symbol.Convolution(data=conv_feat['relu5_3'], kernel=(1, 1), num_filter=256, name="P5_lateral")

    # P5 2x upsampling + C4 = P4
    P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    P4_la = mx.symbol.Convolution(data=conv_feat['relu4_3'], kernel=(1, 1), num_filter=256, name="P4_lateral")
    P4_la_clip = mx.symbol.Crop(*[P4_la, P5_up], name="P4_clip")
    P4 = mx.sym.ElementWiseSum(*[P5_up, P4_la_clip], name="P4_sum")
    P4 = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4_aggregate")

    # P4 2x upsampling + C3 = P3
    P4_up = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P3_la = mx.symbol.Convolution(data=conv_feat['relu3_3'], kernel=(1, 1), num_filter=256, name="P3_lateral")
    P3_la_clip = mx.symbol.Crop(*[P3_la, P4_up], name="P3_clip")
    P3 = mx.sym.ElementWiseSum(*[P4_up, P3_la_clip], name="P3_sum")
    P3 = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_aggregate")

    # # P3 2x upsampling + C2 = P2
    # P3_up   = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    # P2_la   = mx.symbol.Convolution(data=conv_feat[3], kernel=(1, 1), num_filter=256, name="P2_lateral")
    # P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
    # P2      = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
    # P2      = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2_aggregate")

    # # P6 2x subsampling P5
    # P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6_subsampling')

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride16": P5, "stride8": P4, "stride4": P3})
    # conv_fpn_feat.update({"stride64":P6, "stride32":P5, "stride16":P4, "stride8":P3, "stride4":P2})

    return conv_fpn_feat, [P5, P4, P3]


def get_hypernet_feature(conv_fpn_feat):
    relu3_3_down = mx.symbol.Pooling(data=conv_fpn_feat['stride4'], kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                                     pool_type='max', name='relu3_3_subsampling')
    relu5_3_up = mx.symbol.UpSampling(conv_fpn_feat['stride16'], scale=2, sample_type='nearest', workspace=512,
                                      name='relu5_3_upsampling', num_args=1)
    relu4_3 = conv_fpn_feat['stride8']
    relu5_3_up = mx.symbol.pad(data=relu5_3_up, mode='edge', pad_width=(0, 0, 0, 0, 0, 0, 1, 1))
    relu5_3_up = mx.symbol.Crop(*[relu5_3_up, relu4_3], name="C5_clip")
    hyper_feat = mx.symbol.Concat(relu3_3_down, relu4_3, relu5_3_up, dim=1, name="hyper_concat")

    return hyper_feat


def relu2stride(conv_feat):
    result = dict()
    result['stride16'] = conv_feat['relu5_3']
    result['stride8'] = conv_feat['relu4_3']
    result['stride4'] = conv_feat['relu3_3']

    return result


def get_vgg2_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
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
    # hyper feature
    # conv_fpn_feat, _ = get_vgg_conv_dowm(conv_feat)
    # conv_fpn_feat = relu2stride(conv_feat)
    # hyper_feat = get_hypernet_feature(conv_fpn_feat)

    # RPN feature
    # relu5_3 = mx.symbol.Convolution(data=hyper_feat, no_bias=False, kernel=(1, 1), pad=(0, 0), num_filter=512,
    #                                 name="rpn_reduce")
    # relu5_3 = mx.symbol.Activation(data=relu5_3, act_type="relu", name="rpn_reduce_relu")

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
    # rcnn_feat = mx.symbol.Convolution(
    #     data=hyper_feat, kernel=(1, 1), pad=(0, 0), num_filter=512, name='rcnn_reduced')
    # rcnn_feat = mx.symbol.Activation(data=rcnn_feat, act_type="relu", name="rcnn_reduced_relu")
    # pool5 = mx.symbol.ROIPooling(
    #     name='roi_pool5', data=rcnn_feat, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # multi-layer pooling
    # pool5 = get_multi_layer_feature(conv_feat, rois)

    drop6 = get_adaptive_pooling_feature(conv_feat, rois)

    # # attention
    # for r in xrange(1, 2):
    #     fc_hidden = get_fc_net(data=pool5)
    #     pool5 = get_channel_wise_attention(data=pool5, hidden=fc_hidden, n=config.TEST.RPN_POST_NMS_TOP_N)
    # drop7 = get_fc_net(data=pool5)

    # group 6
    # flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    # fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    # relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
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


def get_vgg2_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
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
    # hyper feature
    # conv_fpn_feat, _ = get_vgg_conv_dowm(conv_feat)
    # conv_fpn_feat = relu2stride(conv_feat)
    # hyper_feat = get_hypernet_feature(conv_fpn_feat)

    # RPN feature
    # relu5_3 = mx.symbol.Convolution(data=hyper_feat, no_bias=False, kernel=(1, 1), pad=(0, 0), num_filter=512,
    #                                 name="rpn_reduce")
    # relu5_3 = mx.symbol.Activation(data=relu5_3, act_type="relu", name="rpn_reduce_relu")

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
    # rcnn_feat = mx.symbol.Convolution(
    #     data=hyper_feat, kernel=(1, 1), pad=(0, 0), num_filter=512, name='rcnn_reduced')
    # rcnn_feat = mx.symbol.Activation(data=rcnn_feat, act_type="relu", name="rcnn_reduced_relu")
    # pool5 = mx.symbol.ROIPooling(
    #     name='roi_pool5', data=rcnn_feat, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # multi-layer pooling
    # pool5 = get_multi_layer_feature(conv_feat, rois)

    drop6 = get_adaptive_pooling_feature(conv_feat, rois)

    # attention
    # for r in xrange(1, 2):
    #     fc_hidden = get_fc_net(data=pool5)
    #     pool5 = get_channel_wise_attention(data=pool5, hidden=fc_hidden, n=config.TRAIN.BATCH_ROIS)
    # drop7 = get_fc_net(data=pool5)

    # # group 6
    # flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    # fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    # relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
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
