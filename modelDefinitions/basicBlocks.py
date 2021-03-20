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
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)

def conv9x9(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)
# Middle network of residual dense blocks


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


class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out
# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError

class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out

# DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, 2 * in_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out

# RDB-based RNN cell



class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none', sn = False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


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
        #self.DA = SELayer(nf)
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
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, mFactor= 0.2):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, mFactor=mFactor)
        #self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        #self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        #out = self.RDB2(out)
        #out = self.RDB3(out)
        return out * 0.8 + x



class pixelShuffleUpsampling(nn.Module):
    def __init__(self, inputFilters, scailingFactor=2):
        super(pixelShuffleUpsampling, self).__init__()
    
        self.upSample = nn.Sequential(  nn.Conv2d(inputFilters, inputFilters * (scailingFactor**2), 3, 1, 1),
                                        nn.BatchNorm2d(inputFilters * (scailingFactor**2)),
                                        nn.PixelShuffle(upscale_factor=scailingFactor),
                                        nn.ReLU(inplace=True)
                                    )
    def forward(self, tensor):
        return self.upSample(tensor)

class MKFA(nn.Module):
    def __init__(self, n_feats):
        super(MKFA, self).__init__()

        self.n_feats = n_feats 

        # Convolution blocks
        self.F_B0 = conv3x3(3, self.n_feats, stride=1)
        self.F_B1 = conv3x3(3, self.n_feats, stride=1)

        # F_h: Spatial Transformation
        self.F_S = SEACB(self.n_feats * 2, self.n_feats)

        # Multi Kernel Feature Aggregation
        self.F_M = multiKernelBlock(self.n_feats ) 
        

    def forward(self, x):

        outL = F.leaky_relu(self.F_B0(x))
        outR = F.leaky_relu(self.F_B1(x))

        out = torch.cat([outL, outR], dim=1)
        spatialRelation = self.F_S(out)

        out = self.F_M(spatialRelation)
        return out

class RDBCell(nn.Module):
    def __init__(self, n_feats):
        super(RDBCell, self).__init__()
        self.activation = 'gelu'#para.activation
        self.n_feats = n_feats#para.n_features
        self.n_blocks = 5#para.n_blocks
        self.F_B0 = conv3x3(3, self.n_feats, stride=1)
        self.F_B1 = multiKernelBlock(self.n_feats, self.n_feats)
        self.tempAtten = SELayer(self.n_feats)
        self.F_B2 = conv3x3(self.n_feats , self.n_feats, stride=1) #RDNet(2 * self.n_feats, 2, 3, 3)
        self.sigmoid = torch.nn.Sigmoid()
        # F_h: hidden state part
        self.F_h = SEACB(self.n_feats, self.n_feats)#nn.Sequential(
            #conv3x3(self.n_feats, self.n_feats),
            #RRDB(self.n_feats),#RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation),
            #conv3x3(self.n_feats, self.n_feats)
        #)
        #self.RD = RDB(64, 128,5)

    def forward(self, x):
        #s_last = x
        outL = F.leaky_relu(self.F_B0(x))
        outR = F.leaky_relu(self.F_B1(outL))
        #out = torch.cat([outL, outR], dim=1)
        outAttention = self.tempAtten(outR)
        out = self.F_B2(outAttention)
        
        #
        #out = self.F_h(out)#.mul(outAttention)
        return out + (0.2 * outL) #+ (0.2 * outR)#.mul(outAttention)

from torch import nn

Pool = nn.MaxPool2d

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
from torch import nn

Pool = nn.MaxPool2d

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2


class multiKernelBlock(nn.Module):
    def __init__(self, nf, gc, bias=True):
        super(multiKernelBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.DA = SELayer(nf*4)
        self.conv9 = nn.Conv2d(nf, nf, 9, 1, 4, bias=bias)
        #self.down9 = torch.nn.MaxPool2d(2,2) 
        self.conv7 = nn.Conv2d(nf, nf, 7, 1, 3, bias=bias)#nn.Conv2d(nf + gc, gc, 9, 1, 4, bias=bias)
        #self.down7 = torch.nn.MaxPool2d(2,2) 
        self.conv5 = nn.Conv2d(nf, nf, 5, 1, 2, bias=bias)
        #self.down5 = torch.nn.MaxPool2d(2,2) 
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        #self.down3 = torch.nn.MaxPool2d(2,2)

        self.conv1 = nn.Conv2d(nf * 4, gc, 1, 1, 0)
        #self.down1 = torch.nn.MaxPool2d(2,2) 
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):

        x9 = self.lrelu(self.conv9(x))
        x7 = self.lrelu(self.conv7(x)) + x9
        x5 = self.lrelu(self.conv5(x)) + x9 + x7
        x3 = self.lrelu(self.conv3(x)) + x9 + x7 + x5

        xCat = torch.cat((x9, x7, x5, x3), 1)
        xDA = self.DA(xCat)
        xOut = self.conv1(xDA)
        return xOut * 0.2  #* xDA




class SEACB(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super(SEACB, self).__init__()

        self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=1,
                                     padding=1, bias=False, padding_mode=padding_mode)

        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(3, 1), stride=1,
                                  padding=(1, 0), bias=False, padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(1, 3), stride=1, 
                                  padding=(0, 1), bias=False, padding_mode=padding_mode)

    def forward(self, input):
        square_outputs = self.square_conv(input)
        vertical_outputs = self.ver_conv(input)
        horizontal_outputs = self.hor_conv(input)

        return square_outputs + vertical_outputs + horizontal_outputs

class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.Tensor(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
#net = multiKernelBlock(64, 64)#.cuda()
#summary(net, input_size = (64,128, 128))