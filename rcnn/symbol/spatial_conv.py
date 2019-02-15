# spatial cnn
# implement "Spatial As Deep: Spatial CNN for Traffic Scene Understanding, 201712"

import mxnet as mx


class convDU():

    def __init__(self, in_out_channels=2048, kernel_size=(9, 1)):
        super(convDU, self).__init__()
        self.conv_w = mx.sym.Variable("convDU_weight")
        self.conv_b = mx.sym.Variable("convDU_bias")

        rpn_conv = mx.symbol.Convolution(
            data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")

    def get_DU_feat(self):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in xrange(h):
            i_fea = fea.select(2, i).resize(n, c, 1, w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)