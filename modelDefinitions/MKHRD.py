import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from modelDefinitions.basicBlocks import *    



class ResMKHDR(nn.Module):
    def __init__(self, features = 64, kernel_size = 3,  padding = 1):
        super(ResMKHDR, self).__init__()
        #print("Model 2")
        self.inpConv = nn.Conv2d(3,features,3,1,1)
        self.norm1 =  nn.BatchNorm2d(features)


        block1 = []
        self.block1 = RRDB(features)
        self.attention1 = SELayer(features) # not included in the architectures
        self.attentionSpatial1 = SpatialAttentionBlock(features)
        self.noiseGate1 = nn.Conv2d(features, features, 1,1,0)
       
        block2 = []

        self.block2 = RRDB(features)
        self.noiseGate2 = nn.Conv2d(features, features, 1,1,0)
        self.attention2 = SELayer(features) # not included in the architectures
        self.attentionSpatial2 = SpatialAttentionBlock(features)

        self.convOut = nn.Conv2d(features,3,1,1)

        self.dropoutG = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        
        # Weight Initialization
        #self._initialize_weights()

    def forward(self, img):
    
        #print(affinity.shape)
        xInp = self.inpConv(img) 

        xG = self.block1(xInp) + self.attentionSpatial1(xInp)

        xG = self.block2(xG) + self.attentionSpatial2(xG)

        out = self.relu(self.convOut(xG) + img)

        return out #, outUp'''

    
    def _initialize_weights(self):

        #self.inputConvLeft.apply(init_weights)
        self.inpConv.apply(init_weights)
        self.block1.apply(init_weights)
        self.block2.apply(init_weights)

         
        self.convOut.apply(init_weights)


class HDRRangeNet(nn.Module):
    def __init__(self, features = 64, kernel_size = 3,  padding = 1):
        super(HDRRangeNet, self).__init__()
        #print("Model 2")
        self.inpConv = nn.Conv2d(3,features,3,1,1)
        self.norm1 =  nn.BatchNorm2d(features)

        self.blockG = RRDB(features, mFactor=0.5)
        self.attention = SELayer(features) # not included in the architectures
        self.attentionSpatial = SpatialAttentionBlock(features)
        self.noiseGate1 = nn.Conv2d(features, features, 1,1,0)

        self.block1 = RRDB(features, mFactor=0.5) # not included in the architectures
 

        self.convOut = nn.Conv2d(features,3,1,1)
        
        self.dropoutG = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        
        # Weight Initialization
        #self._initialize_weights()

    def forward(self, img):

        xInp = self.inpConv(img) 

        xG = self.blockG(xInp) + self.attentionSpatial(xInp)
        
        out = self.relu(self.convOut(xG) + img)

        return out 

    
    def _initialize_weights(self):

        #self.inputConvLeft.apply(init_weights)
        self.inpConv.apply(init_weights)
        self.blockG.apply(init_weights)
        self.convOut.apply(init_weights)

#net = ResMKHDR()#.cuda()
#summary(net, input_size = (3, 128, 128))
#from ptflops import get_model_complexity_info
#macs, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True,
#                                          print_per_layer_stat=True, verbose=True)
#print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#print('{:<30}  {:<8}'.format('Number of parameters: ', params))