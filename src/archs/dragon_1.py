from constants import *
softmax = nn.Softmax(dim=2)

def mid_conv_layer(depth_in, depth_out, kernel_size=1):
    if kernel_size == 1:
        return nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU(),
                )
    elif kernel_size == 3:
        return nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU(),
                )

def deconv_layer(depth_in, depth_out, output_size=-1):
    if output_size == -1:
        return nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
            )
    else:
        return nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU(),
                    nn.Upsample(size=output_size, mode='bilinear'),
            )

def top_deconv_layer(depth_in, depth_out, output_size=-1):    
    if output_size == -1:
        return nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
            )
    else:
        return nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                    nn.Upsample(size=output_size, mode='bilinear'),
            )

def deconv_fpn_layer(depth_in=256, depth_out=128, num_layers=1):
    modules = []
    for i in range(num_layers):
        if i < num_layers - 1:
            depth_in_, depth_out_ = depth_in, depth_in
        else:
            depth_in_, depth_out_ = depth_in, depth_out
        modules.append(nn.Conv2d(depth_in_, depth_out_, kernel_size=3, stride=1, padding=1))
        modules.append(nn.BatchNorm2d(depth_out_))
        modules.append(nn.ReLU(),)
        modules.append(nn.Upsample(scale_factor=2, mode='bilinear'))
    
    if num_layers == 0:
        modules.append(nn.Conv2d(depth_in, depth_out, kernel_size=3, stride=1, padding=1))
    return nn.Sequential(*modules)

#FPN-style encoder-deconder network
class Arch(nn.Module):
    def __init__(self, output_depth=3):
        super(Arch, self).__init__()
        self.name = 'DRAGON_v1'
        #resnet hyper params
        RESNET_DEPTH_2 = 256
        RESNET_DEPTH_3 = 512
        RESNET_DEPTH_4 = 1024
        RESNET_DEPTH_5 = 2048
        MIDDLE_DEPTH = 128
        DECONV_DEPTH = 32
        self.output_depth = output_depth

        resnet_model = torchvision.models.resnet50(pretrained=True)

        self.conv12 = nn.Sequential(*list(resnet_model.children())[0:5]) #0
        self.conv3 = nn.Sequential(*list(resnet_model.children())[5]) #0
        self.conv4 = nn.Sequential(*list(resnet_model.children())[6]) #1
        self.conv5 = nn.Sequential(*list(resnet_model.children())[7]) #2

        self.mid2 = mid_conv_layer(RESNET_DEPTH_2, MIDDLE_DEPTH) #2
        self.mid3 = mid_conv_layer(RESNET_DEPTH_3, MIDDLE_DEPTH) #3
        self.mid4 = mid_conv_layer(RESNET_DEPTH_4, MIDDLE_DEPTH) #4
        self.mid5 = mid_conv_layer(RESNET_DEPTH_5, MIDDLE_DEPTH) #5

        self.concat_4 = mid_conv_layer(MIDDLE_DEPTH * 2, MIDDLE_DEPTH, 3) #7
        self.concat_3 = mid_conv_layer(MIDDLE_DEPTH * 2, MIDDLE_DEPTH, 3) #8
        self.concat_2 = mid_conv_layer(MIDDLE_DEPTH * 2, MIDDLE_DEPTH, 3) #8

        self.deconv_5 = nn.Upsample(scale_factor=2, mode='bilinear') #6
        self.deconv_4 = nn.Upsample(scale_factor=2, mode='bilinear') #7
        self.deconv_3 = nn.Upsample(scale_factor=2, mode='bilinear') #8
        self.deconv_2 = nn.Upsample(scale_factor=2, mode='bilinear') #8

        self.deconv_fpn_5 = deconv_fpn_layer(depth_in=MIDDLE_DEPTH, depth_out=DECONV_DEPTH, num_layers=3)
        self.deconv_fpn_4 = deconv_fpn_layer(depth_in=MIDDLE_DEPTH, depth_out=DECONV_DEPTH, num_layers=2)
        self.deconv_fpn_3 = deconv_fpn_layer(depth_in=MIDDLE_DEPTH, depth_out=DECONV_DEPTH, num_layers=1)
        self.deconv_fpn_2 = deconv_fpn_layer(depth_in=MIDDLE_DEPTH, depth_out=DECONV_DEPTH, num_layers=0)

        self.top_deconv_layer = top_deconv_layer(DECONV_DEPTH, output_depth, output_size=512)


    def forward(self, x):
        #convolution layers
        conv_out2 = self.conv12(x)

        mid_out2 = self.mid2(conv_out2)
        conv_out3 = self.conv3(conv_out2)

        mid_out3 = self.mid3(conv_out3)
        conv_out4 = self.conv4(conv_out3)

        mid_out4 = self.mid4(conv_out4)
        conv_out5 = self.conv5(conv_out4)

        feature_map5 = self.mid5(conv_out5)
        
        #deconvolution layers
        deconv_out5 = self.deconv_5(feature_map5)        
        fused4 = torch.cat([deconv_out5, mid_out4], 1)
        feature_map4 = self.concat_4(fused4)

        deconv_out4 = self.deconv_4(feature_map4)        
        fused3 = torch.cat([deconv_out4, mid_out3], 1)
        feature_map3 = self.concat_3(fused3)

        deconv_out3 = self.deconv_3(feature_map3)        
        fused2 = torch.cat([deconv_out3, mid_out2], 1)
        feature_map2 = self.concat_2(fused2)

        #FPN
        fpn_map_5 = self.deconv_fpn_5(feature_map5)
        fpn_map_4 = self.deconv_fpn_4(feature_map4)
        fpn_map_3 = self.deconv_fpn_3(feature_map3)
        fpn_map_2 = self.deconv_fpn_2(feature_map2)
        fpn_map = fpn_map_2 + fpn_map_3 + fpn_map_4 + fpn_map_5
        output = self.top_deconv_layer(fpn_map)

        return output


    