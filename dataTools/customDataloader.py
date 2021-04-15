import glob
import numpy as np
import time
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utilities.customUtils import *
from dataTools.dataNormalization import *
from dataTools.customTransform import *
import os
import imgaug.augmenters as iaa
from skimage import io, color
import numpy as np
from skimage import feature
from skimage.transform import rescale, resize, downscale_local_mean
import random
from torch.autograd import Variable


def imread_uint16_png(image_path, alignratio_path):
    """ This function loads a uint16 png image from the specified path and restore its original image range with
    the ratio stored in the specified alignratio.npy respective path.


    Args:
        image_path (str): Path to the uint16 png image
        alignratio_path (str): Path to the alignratio.npy file corresponding to the image

    Returns:
        np.ndarray (np.float32, (h,w,3)): Returns the RGB HDR image specified in image_path.

    """
    # Load the align_ratio variable and ensure is in np.float32 precision
    align_ratio = np.load(alignratio_path).astype(np.float32)
    # Load image without changing bit depth and normalize by align ratio
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio




class customDatasetReader(Dataset):
    def __init__(self, image_list, height, width, transformation=True):
        self.image_list = image_list
        self.transformLR = transforms
        self.imageH = height
        self.imageW = width
        normalize = transforms.Normalize(normMean, normStd)
    
        self.transformRI = transforms.Compose([ 
                                                transforms.ToTensor(),
                                            ])
        self.transformWN = transforms.Compose([ #transforms.Resize((self.imageH, self.imageW)),
                                                transforms.ToTensor(),
                                            
                                            ])

    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):


        gamma = 2.24
        self.sampledImageLeft = cv2.cvtColor(cv2.imread(self.image_list[i], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)/255.0
        
       
        self.gtImageFileName = self.image_list[i].replace('_medium', '_gt16') #+ ".png"
        alignPath =  self.gtImageFileName.replace("_gt16.png", "_alignratio.npy")
        #print(alignPath)
        self.gtImage = imread_uint16_png(self.gtImageFileName, alignPath)
        
        HDRGt8 = self.gtImageFileName.replace("gt16", "gt8")#self.imagePathGT.replace("XtrasHD2/HDRTrainingUnNorm", "XtrasHD1/HDRTrainingNorm") + extractFileName(self.image_list[i],True).replace('_medium', '_gt') + ".png"
        #print(HDRGt8)
        self.sampledImageHDR8 =  cv2.cvtColor(cv2.imread(HDRGt8, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)/255.0
        
        
        randH = random.randint(0, self.imageH/2)
        randW = random.randint(0, self.imageH/2)
        #print(randH, randW)
        self.sampledImageLeft = self.sampledImageLeft[randH:randH + self.imageH, randW:randW + self.imageW, : ]
        self.gtImage = self.gtImage[randH:randH + self.imageH, randW:randW + self.imageW, : ]
        self.sampledImageHDR8 = self.sampledImageHDR8[randH:randH + self.imageH, randW:randW + self.imageW, : ]


        self.sampledImageLeft = cv2.resize(self.sampledImageLeft, (128, 128)).astype(np.float32) #** gamma
        self.gtImage =  cv2.resize(self.gtImage, (128, 128)).astype(np.float32) #** gamma
        self.sampledImageHDR8 =  cv2.resize(self.sampledImageHDR8, (128, 128)).astype(np.float32) #** gamma

        # Transforms Images for training 
        self.inputImageCrop = self.transformRI(self.sampledImageLeft)
        self.gtImageCrop = self.transformWN(self.gtImage)#self.transformRI(self.gtImageCrop)
        self.gt8bitCrop = self.transformWN(self.sampledImageHDR8)

        #print (self.gtImageHR.max(), self.gtImageHR.min(), self.inputImage.max(), self.inputImage.min())
        #print(self.lumImageCrop.shape)
        #print(self.inputImageCrop.shape, self.gtImageCrop.shape)
        #self.gtImageCrop = torch.clamp(self.gtImageCrop, 0, 2.5)
        return self.inputImageCrop , self.gt8bitCrop, self.gtImageCrop
