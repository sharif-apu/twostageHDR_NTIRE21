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
    def __init__(self, image_list, imagePathGT, height, width, transformation=True):
        self.image_list = image_list
        self.imagePathGT = imagePathGT
        self.transformLR = transforms
        self.imageH = height
        self.imageW = width
        normalize = transforms.Normalize(normMean, normStd)

        #self.transformHRGT = transforms.Compose([ transforms.Resize((self.imageH, self.imageW)),
        #                                        transforms.ToTensor(),
        #                                       normalize,
        #                                        ])

    
        self.transformRI = transforms.Compose([ #transforms.Resize((self.imageH, self.imageW)),
                                                transforms.ToTensor(),
                                                #normalize,
                                                #AddGaussianNoise(pov=1.0)
                                            ])
        self.transformWN = transforms.Compose([ #transforms.Resize((self.imageH, self.imageW)),
                                                transforms.ToTensor(),
                                                #normalize,
                                                #AddGaussianNoise(pov=0.8)
                                            ])

    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):

        # Read Images
        #self.sampledImageLeft = cv2.imread(self.image_list[i]
        self.sampledImageLum =  cv2.cvtColor(cv2.imread(self.image_list[i], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2LAB)/255.0
        self.sampledImageLum = self.sampledImageLum[:,:, 0:1]
        self.sampledImageLum = np.concatenate((self.sampledImageLum, self.sampledImageLum, self.sampledImageLum), axis=2).astype(np.float32)
        #print(self.sampledImageLum.shape)
        self.sampledImageLeft = cv2.cvtColor(cv2.imread(self.image_list[i], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)/255.0
        
       
        self.gtImageFileName = self.imagePathGT + extractFileName(self.image_list[i],True).replace('_medium', '_gt') + ".png"
        alignPath =  self.gtImageFileName.replace("gt.png", "alignratio.npy")
        self.gtImage = imread_uint16_png(self.gtImageFileName, alignPath)
        

        randomNum = random.randint(0,100) 
        if randomNum % 1 == 0:
            #print("True", self.sampledImageLeft.shape)
            width, height, ch = self.sampledImageLeft.shape
            #print(width, height, ch)
            randLeft = random.randint(0,width/2)
            randRight = random.randint(0,height/2)

            #print(randRight, randLeft)
            self.sampledImageCrop = self.sampledImageLeft[randLeft : randLeft + 512, randRight : randRight + 512]
            self.gtImageCrop = self.gtImage[randLeft : randLeft + 512, randRight : randRight + 512]
            self.sampledImageLumCrop = self.sampledImageLum[randLeft : randLeft + 512, randRight : randRight + 512]
            #print(self.sampledImageCrop.shape, self.gtImageCrop.shape)

        self.sampledImageLeft = cv2.resize(self.sampledImageCrop, (128, 128)).astype(np.float32)
        self.gtImage =  cv2.resize(self.gtImageCrop, (128, 128)).astype(np.float32)
        self.sampledImageLum =  cv2.resize(self.sampledImageLumCrop, (128, 128)).astype(np.float32)

        # Transforms Images for training 
        self.inputImageCrop = self.transformRI(self.sampledImageLeft)
        self.gtImageCrop = self.transformWN(self.gtImage)#self.transformRI(self.gtImageCrop)
        self.lumImageCrop = self.transformRI(self.sampledImageLum)

        #print (self.gtImageHR.max(), self.gtImageHR.min(), self.inputImage.max(), self.inputImage.min())
        #print(self.lumImageCrop.shape)
        #print(self.inputImageCrop.shape, self.gtImageCrop.shape)
        return self.inputImageCrop, self.lumImageCrop, self.gtImageCrop
