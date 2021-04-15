import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torchsummary import summary

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def swish(x):
    return x * torch.sigmoid(x)




class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SpatialAttentionBlock(nn.Module):
    def __init__(self, spatial_filter=32):
        super(SpatialAttentionBlock, self).__init__()
        self.spatialAttenton = SpatialAttention()
        self.conv = nn.Conv2d(spatial_filter, spatial_filter,  3, padding=1)


    def forward(self, x):
        x1 = self.spatialAttenton(x)
        #print(" spatial attention",x1.shape)
        xC = self.conv(x)
        #print("conv",xC.shape)
        y = x1 * xC
        #print("output",y.shape)
        return y
        
class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32,mFactor = 0.2, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.mFactor = mFactor
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.ReLU()

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        #xDA = self.DA(x)
        x1 = self.lrelu(self.conv1(x))
        #print(x1.shape)
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        #print(x2.shape)
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        #print(x3.shape)
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        #print(x4.shape)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.mFactor + x #+ xDA


class RRDB(nn.Module):


    def __init__(self, nf, gc=32, mFactor= 0.2):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, mFactor=mFactor)


    def forward(self, x):
        out = self.RDB1(x)
        return out * 0.8 + x
#net = multiKernelBlock(64, 64)#.cuda()
#summary(net, input_size = (64,128, 128))