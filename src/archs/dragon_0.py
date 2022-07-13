from constants import *
softmax = nn.Softmax(dim=2)

def mid_conv_layer(depth_in, depth_out):
    return nn.Sequential(
                nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
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

class Arch(nn.Module):
    def __init__(self, output_depth=3):
        super(Arch, self).__init__()
        self.name = 'DRAGON_v0'
        #resnet hyper params
        RESNET_DEPTH_3 = 512
        RESNET_DEPTH_4 = 1024
        RESNET_DEPTH_5 = 2048
        self.output_depth = output_depth

        resnet_model = torchvision.models.resnet50(pretrained=True)

        self.conv123 = nn.Sequential(*list(resnet_model.children())[0:6]) #0
        self.conv4 = nn.Sequential(*list(resnet_model.children())[6]) #1
        self.conv5 = nn.Sequential(*list(resnet_model.children())[7]) #2

        self.mid3 = mid_conv_layer(RESNET_DEPTH_3, output_depth) #3
        self.mid4 = mid_conv_layer(RESNET_DEPTH_4, output_depth) #4
        self.mid5 = mid_conv_layer(RESNET_DEPTH_5, output_depth)  #child 5

        self.deconv_5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear')) #child 6
        self.deconv_4 = deconv_layer(output_depth * 2, output_depth) #7
        self.deconv_3 = deconv_layer(output_depth * 2, output_depth, output_size=512) #8


    def forward(self, x):
        #convolution layers
        conv_out3 = self.conv123(x)
        mid_out3 = self.mid3(conv_out3)
        conv_out4 = self.conv4(conv_out3)
        mid_out4 = self.mid4(conv_out4)
        conv_out5 = self.conv5(conv_out4)
        mid_out5 = self.mid5(conv_out5)

        #deconvolution layers
        deconv_out5 = self.deconv_5(mid_out5)
        
        fused4 = torch.cat([deconv_out5, mid_out4], 1)
        deconv_out4 = self.deconv_4(fused4)

        fused3 = torch.cat([deconv_out4, mid_out3], 1)
        output = self.deconv_3(fused3)

        return output


    