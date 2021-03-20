import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from modelDefinitions.basicBlocks import *    



class ResMKHDR(nn.Module):
    def __init__(self, features = 64, kernel_size = 3,  padding = 1):
        super(ResMKHDR, self).__init__()
        #print("Model 2")
        self.inpConv = nn.Conv2d(3,features,3,1,1)#RDBCell(features)
        #self.inpLum = nn.Conv2d(3,features,3,1,1)
        #self.mk1 =  multiKernelBlock(features, features)
        self.norm1 =  nn.BatchNorm2d(features)


        block1 = []
        self.block1 = RRDB(features)#nn.Sequential(*block1)
        self.attention1 = SELayer(features)
        self.attentionSpatial1 = SpatialAttentionBlock(features)
        self.noiseGate1 = nn.Conv2d(features, features, 1,1,0)
       
        block2 = []

        self.block2 = RRDB(features)
        self.noiseGate2 = nn.Conv2d(features, features, 1,1,0)
        self.attention2 = SELayer(features)
        self.attentionSpatial2 = SpatialAttentionBlock(features)

        self.convOut = nn.Conv2d(features,3,1,1)
        #self.outUp = pixelShuffleUpsampling(inputFilters=3, scailingFactor=2)
        
        self.dropoutG = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        
        # Weight Initialization
        #self._initialize_weights()

    def forward(self, img):
    
        #print(affinity.shape)
        xInp = self.inpConv(img) 
        #xLum = self.inpLum(imgLum)
        #xL = self.blockL(xLum) + xLum
        
        xG = self.block1(xInp) + self.attentionSpatial1(xInp)
        #xG = self.attention(xG) 
        #xG = self.attentionSpatial1(xG)

        xG = self.block2(xG) + self.attentionSpatial2(xG)
        #xG = self.attention1(xG) 
        #xG = self.attentionSpatial2(xG) 
        
        #x1 = self.attentionSpatial1(self.attention1(self.block1(xInp) +  self.noiseGate1(xInp) )) 
        #x2 = self.attentionSpatial2(self.attention2(self.block2(x1) + self.noiseGate2(x1))) 

        out = self.relu(self.convOut(xG) + img)

        return out #, outUp'''

    
    def _initialize_weights(self):

        #self.inputConvLeft.apply(init_weights)
        self.inpConv.apply(init_weights)
        
        #self.blockL.apply(init_weights)
        self.block1.apply(init_weights)
        self.block2.apply(init_weights)
        #self.blockG.apply(init_weights)

         
        self.convOut.apply(init_weights)


class HDRRangeNet(nn.Module):
    def __init__(self, features = 64, kernel_size = 3,  padding = 1):
        super(HDRRangeNet, self).__init__()
        #print("Model 2")
        self.inpConv = nn.Conv2d(3,features,3,1,1)
        #self.inpLum = nn.Conv2d(3,features,3,1,1)
        #self.mk1 =  multiKernelBlock(features, features)
        self.norm1 =  nn.BatchNorm2d(features)

        self.blockG = RRDB(features, mFactor=0.5)#nn.Sequential(*blockG)
        self.attention = SELayer(features)
        self.attentionSpatial = SpatialAttentionBlock(features)
        self.noiseGate1 = nn.Conv2d(features, features, 1,1,0)

        self.block1 = RRDB(features, mFactor=0.5)#nn.Sequential(*blockG)
        #self.attention1 = SELayer(features)
        #self.attentionSpatial1 = SpatialAttentionBlock(features)
        #self.noiseGate2 = nn.Conv2d(features, features, 1,1,0)


        self.convOut = nn.Conv2d(features,3,1,1)
        #self.outUp = pixelShuffleUpsampling(inputFilters=3, scailingFactor=2)
        
        self.dropoutG = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        
        # Weight Initialization
        #self._initialize_weights()

    def forward(self, img):
    
        #print(affinity.shape)
        xInp = self.inpConv(img) 

        #xInp = self.attentionSpatial(xInp)
        xG = self.blockG(xInp) + self.attentionSpatial(xInp)
        #xG = self.attention(xG) 
        #xG = self.relu()

        #xG = self.block1(xG) + xG#+ self.noiseGate2(xG)
        #xG = self.attention1(xG) 
        #xG = self.attentionSpatial1(xG) 
        
        
        out = self.relu(self.convOut(xG) + img)

        return out #, outUp'''

    
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