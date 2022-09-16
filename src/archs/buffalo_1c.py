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
    def __init__(self, output_depth=3, skip_connection=True):
        super(Arch, self).__init__()
        self.name = 'BUFFALO_v1'
        #resnet hyper params
        self.HIDDEN_SIZE_0 = 1024
        self.NUM_LAYERS_0 = 1
        self.INPUT_LSTM_0 = 1024
        self.GLOBAL_FEATURE_SIZE = 2048

        # self.HIDDEN_SIZE_1 = 512
        # self.NUM_LAYERS_1 = 2
        # self.INPUT_LSTM_1 = 256

        # self.HIDDEN_SIZE_2 = 2048
        # self.NUM_LAYERS_2 = 1
        # self.INPUT_LSTM_2 = 512

        MIDDLE_DEPTH = 256
        DECONV_DEPTH = 32
        self.output_depth = output_depth
        self.skip_connection = skip_connection

        self.lstm_0 = nn.LSTM(input_size=self.INPUT_LSTM_0, hidden_size =self.HIDDEN_SIZE_0, num_layers =self.NUM_LAYERS_0, 
                            batch_first=True, bidirectional=False)
        
        self.linear = nn.Linear(self.HIDDEN_SIZE_0, self.GLOBAL_FEATURE_SIZE)

        # self.lstm_1 = nn.LSTM(input_size=self.INPUT_LSTM_1, hidden_size =self.HIDDEN_SIZE_1, num_layers =self.NUM_LAYERS_1, 
        #                     batch_first=True, bidirectional=False)

        # self.lstm_2 = nn.LSTM(input_size=self.INPUT_LSTM_2, hidden_size =self.HIDDEN_SIZE_2, num_layers =self.NUM_LAYERS_2, 
        #                     batch_first=True, bidirectional=False)

        # self.mid_init = mid_conv_layer(depth_in=self.HIDDEN_SIZE_0, depth_out=MIDDLE_DEPTH)
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
        # self.AvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, binary_encode):
        # print ('binary_encode', binary_encode.size())
        h0 = torch.zeros(self.NUM_LAYERS_0, binary_encode.size(0), self.HIDDEN_SIZE_0).to(device)
        c0 = torch.zeros(self.NUM_LAYERS_0, binary_encode.size(0), self.HIDDEN_SIZE_0).to(device)
        lstm_0, _ = self.lstm_0(binary_encode, (h0, c0))

        # h1 = torch.zeros(self.NUM_LAYERS_1, binary_encode.size(0), self.HIDDEN_SIZE_1).to(device)
        # c1 = torch.zeros(self.NUM_LAYERS_1, binary_encode.size(0), self.HIDDEN_SIZE_1).to(device)
        # lstm_1, _ = self.lstm_1(lstm_0, (h1, c1))

        # h2 = torch.zeros(self.NUM_LAYERS_2, binary_encode.size(0), self.HIDDEN_SIZE_2).to(device)
        # c2 = torch.zeros(self.NUM_LAYERS_2, binary_encode.size(0), self.HIDDEN_SIZE_2).to(device)
        # lstm_2, _ = self.lstm_2(lstm_1, (h2, c2))


        lstm_out = lstm_0[:, -1, :] #B, hidden size
        # print ('lstm_out', lstm_out.size())
        # relu = nn.ReLU()
        # lstm_out = relu(lstm_out)
        global_feature = self.linear(lstm_out)
        # global_feature = torch.mean(lstm_0, axis=1)
        # feature_map5 = torch.unsqueeze(global_feature, 2)
        # feature_map5 = torch.unsqueeze(feature_map5, 3)
        # feature_map5 = self.mid_init(feature_map5)
        # feature_map5 = self.deconv_init(feature_map5)
        # # print ('encode feature_map5', feature_map5.size())

        # #deconvolution layers
        # feature_map4 = self.deconv_5(feature_map5)        
        # feature_map3 = self.deconv_4(feature_map4)        
        # feature_map2 = self.deconv_3(feature_map3)        

        # #FPN
        # fpn_map_5 = self.deconv_fpn_5(feature_map5)
        # fpn_map_4 = self.deconv_fpn_4(feature_map4)
        # fpn_map_3 = self.deconv_fpn_3(feature_map3)
        # fpn_map_2 = self.deconv_fpn_2(feature_map2)
        # fpn_map = fpn_map_2 + fpn_map_3 + fpn_map_4 + fpn_map_5
        # output = self.top_deconv_layer(fpn_map)

        # # global_feature = self.AvgPool2d(feature_map5)
        # global_feature = torch.squeeze(global_feature)

        return None, global_feature


    