from constants import *
softmax = nn.Softmax(dim=2)


#FPN-style encoder-deconder network
class Arch(nn.Module):
    def __init__(self, output_depth=3, skip_connection=True):
        super(Arch, self).__init__()
        self.name = 'BUFFALO_v1'
        
        self.encode = nn.Sequential(nn.Linear(524288, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 512),
                                )

        self.decode = nn.Sequential(nn.Linear(512, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 524288),
                                )

        # self.mid_init = mid_conv_layer(depth_in=self.HIDDEN_SIZE_2, depth_out=MIDDLE_DEPTH)
        # self.deconv_init = nn.Upsample(scale_factor=2, mode='bilinear') #6
        # self.deconv_5 = nn.Upsample(scale_factor=2, mode='bilinear') #6
        # self.deconv_4 = nn.Upsample(scale_factor=2, mode='bilinear') #7
        # self.deconv_3 = nn.Upsample(scale_factor=2, mode='bilinear') #8
        # self.deconv_2 = nn.Upsample(scale_factor=2, mode='bilinear') #8

        # self.deconv_fpn_5 = deconv_fpn_layer(depth_in=MIDDLE_DEPTH, depth_out=DECONV_DEPTH, num_layers=3)
        # self.deconv_fpn_4 = deconv_fpn_layer(depth_in=MIDDLE_DEPTH, depth_out=DECONV_DEPTH, num_layers=2)
        # self.deconv_fpn_3 = deconv_fpn_layer(depth_in=MIDDLE_DEPTH, depth_out=DECONV_DEPTH, num_layers=1)
        # self.deconv_fpn_2 = deconv_fpn_layer(depth_in=MIDDLE_DEPTH, depth_out=DECONV_DEPTH, num_layers=0)

        # self.top_deconv_layer = top_deconv_layer(DECONV_DEPTH, output_depth, output_size=64)
        

    def forward(self, binary_encode):
        # print ('binary_encode', binary_encode.size())
        encode = self.encode(binary_encode)
        decode = self.decode(encode)
        return decode, encode


    