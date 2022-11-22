from constants import *
softmax = nn.Softmax(dim=2)


# 1x1 convolution
def conv1(in_channels, out_channels, stride=1):
    return [nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)]
    

# 3x3 convolution
def conv3(in_channels, out_channels, stride=1):
    return [nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)]

#FPN-style encoder-deconder network
class Arch(nn.Module):
    def __init__(self, output_depth=3, skip_connection=True):
        super(Arch, self).__init__()
        self.name = 'BUFFALO_v1'
        #resnet hyper params
        self.HIDDEN_SIZE_0 = 2048
        self.NUM_LAYERS_0 = 1
        self.INPUT_LSTM_0 = 1024

        self.conv_4 = nn.Sequential(*(conv3(1, 4) + conv1(4, 4) + conv3(4, 8, 2)))
        N = 8
        self.conv_8 = nn.Sequential(*(conv3(N, N, 2) + conv3(N, N) + conv1(N, N) + conv3(N, N*2, 2)))
        N *= 2
        self.conv_16 = nn.Sequential(*(conv3(N, N, 2) + conv3(N, N) + conv1(N, N) + conv3(N, N*2, 2)))
        N *= 2
        self.conv_32 = nn.Sequential(*(conv3(N, N, 2) + conv3(N, N) + conv1(N, N) + conv3(N, N*2, 2)))
        N *= 2
        self.conv_64 = nn.Sequential(*(conv3(N, N, 2) + conv3(N, N) + conv1(N, N) + conv3(N, N*2, 2)))
        N *= 2
        self.conv_128 = nn.Sequential(*(conv3(N, N, 2) + conv3(N, N) + conv1(N, N) + conv3(N, N*2, 2)))
        N *= 2
        self.conv_256 = nn.Sequential(*(conv3(N, N, 2) + conv3(N, N) + conv1(N, N) + conv3(N, N*2, 2)))
        N *= 2
        self.conv_512 = nn.Sequential(*(conv3(N, N, 2) + conv3(N, N) + conv1(N, N) + conv3(N, N*2, 2)))
        N *= 2
        self.conv_1024 = nn.Sequential(*(conv3(N, N, 2) + conv3(N, N) + conv1(N, N)))
        self.conv_last = nn.Conv1d(N, N*2, kernel_size=1, stride=1, padding=0, bias=False)
                                        
                                        
        self.AvgPool1d = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, binary_encode):
        # print ('binary_encode', binary_encode.size())

        out = self.conv_4(binary_encode)
        # print ('conv_4', out.size())
        out = self.conv_8(out)
        # print ('conv_8', out.size())
        out = self.conv_16(out)
        # print ('conv_16', out.size())
        out = self.conv_32(out)
        # print ('conv_32', out.size())
        out = self.conv_64(out)
        # print ('conv_64', out.size())
        out = self.conv_128(out)
        # print ('conv_128', out.size())
        out = self.conv_256(out)
        # print ('conv_256', out.size())
        out = self.conv_512(out)
        # print ('conv_512', out.size())
        out = self.conv_1024(out)
        # print ('conv_1024', out.size())
        out = self.conv_last(out)


        # out_resnet = self.resnet_sequence(binary_encode)
        # print ('out_resnet', out_resnet.size())

        global_feature = self.AvgPool1d(out)
        # print ('global_feature', global_feature.size())
        global_feature = torch.squeeze(global_feature)


        return None, global_feature


    