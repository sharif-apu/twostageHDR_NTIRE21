import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os 
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
from utilities.customUtils import *
from dataTools.sampler import *
import numpy as np
import cv2
from PIL import Image
from dataTools.dataNormalization import *
import skimage.io as io

def torchTensorToNumpy( image):
        imageNP = image.cpu().detach().numpy()#.reshape(image.shape[1], image.shape[2], image.shape[0])
        return imageNP
def imwrite_uint16_png(image_path, image, alignratio_path):
    """ This function writes the hdr image as a uint16 png and stores its related align_ratio value in the specified paths.

        Args:
            image_path (str): Write path to the uint16 png image (needs to finish in .png, e.g. 0000.png)
            image (np.ndarray): HDR image in float format.
            alignratio_path (str): Write path to the align_ratio value (needs to finish in .npy, e.g. 0000_alignratio.npy)

        Returns:
            np.ndarray (np.float32, (h,w,3)): Returns the RGB HDR image specified in image_path.

    """
    align_ratio = (2 ** 16 - 1) / image.max()
    np.save(alignratio_path, align_ratio)
    uint16_image_gt = np.round(image * align_ratio).astype(np.uint16)
    cv2.imwrite(image_path, cv2.cvtColor(uint16_image_gt, cv2.COLOR_RGB2BGR))
    return None
class AddGaussianNoise(object):
    def __init__(self, noiseLevel):
        self.var = 0.1
        self.mean = 0.0
        self.noiseLevel = noiseLevel
        
    def __call__(self, tensor):
        sigma = self.noiseLevel/255
        noisyTensor = tensor + torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor 
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)


class inference():
    def __init__(self, inputRootDir, outputRootDir, modelName, resize = None, validation = None ):
        self.inputRootDir = inputRootDir
        self.outputRootDir = outputRootDir
        self.modelName = modelName
        self.resize = resize
        self.validation = validation
        self.unNormalize = UnNormalize()
    


    def inputForInference(self, imagePath, noiseLevel):
        
       
        #print(imagePath, imagePath.replace("_l", "_r"))
        imgL = cv2.cvtColor(cv2.imread(imagePath, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)/255.0
        imgL = imgL.astype(np.float32)

        lumImg =  cv2.cvtColor(cv2.imread(imagePath, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2LAB)/255.0
        lumImg = lumImg[:,:, 0:1]
        lumImg = np.concatenate((lumImg, lumImg, lumImg), axis=2).astype(np.float32)
        #print(lumImg.shape)
        #print(type(imgL), imagePath)
        device = torch.device('cuda:0')
        transform = transforms.Compose([#transforms.Resize((512, 512)), 
                                        transforms.ToTensor(),
                                        #transforms.Normalize(normMean, normStd),
                                        ])
        transformD = transforms.Compose([#transforms.Resize((512, 512)),
                                        transforms.ToTensor(),
                                        #transforms.Normalize(normMean, normStd),
                                        ])

        testImgL = transform(imgL).unsqueeze(0)
        testLumL = transform(lumImg).unsqueeze(0)
        #testImgLD = transformD(imgLD).unsqueeze(0)
        #testImgR = transform(imgR).unsqueeze(0)
        #testImgRD = transformD(imgRD).unsqueeze(0)
        #print("input",imagePath,self.unNormalize(testImg).max(), self.unNormalize(testImg).min())
        return testImgL, testLumL#, testImgLD, testImgR, testImgRD
        

    def saveModelOutput(self, modelOutput, inputImagePath, step = None, ext = ".png"):
        datasetName = inputImagePath.split("/")[-2]
        if step:
        

            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True)  + \
                              "_" + str(step) + ext
            save_image(modelOutput[0], imageSavingPath)
            
        else:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True) + ext
            imageSavingPath16bit = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True).split("_")[-2] + ext
            alignRationPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True).split("_")[-2]  + '_alignratio.npy'
            imgSq = modelOutput.squeeze(0).cpu().numpy()
            imgReshape =  np.transpose(imgSq,(1,2,0))
            imwrite_uint16_png(imageSavingPath16bit, imgReshape, alignRationPath)
            #save_image(modelOutput[0], imageSavingPath)
        
        #save_image(self.unNormalize(modelOutput[0]), imageSavingPath)
        #print(imageSavingPath)
        #print(inputImagePath,self.unNormalize(modelOutput[0]).max(), self.unNormalize(modelOutput[0]).min())

    

    def testingSetProcessor(self):
        testSets = glob.glob(self.inputRootDir+"*/")
        #print ("DirPath",self.inputRootDir+"*/")
        if self.validation:
            #print(self.validation)
            testSets = testSets[:1]
        #print (testSets)
        testImageList = []
        for t in testSets:
            testSetName = t.split("/")[-2]
            #print("Dir Path",self.outputRootDir + self.modelName  + "/" + testSetName )
            createDir(self.outputRootDir + self.modelName  + "/" + testSetName)
            imgInTargetDir = imageList(t, False)
            testImageList += imgInTargetDir

        return testImageList


