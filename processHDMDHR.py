import glob
import os  
import shutil  
from pathlib import Path
import ntpath
import cv2
import numpy as np
import png
import cv2
import shutil 
import random
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
from datetime import datetime
import argparse
import sys

def mainParser(args=sys.argv[1:]):

    # Parser definition
    parser = argparse.ArgumentParser(description="Parses command.")

    # Parser Options
    parser.add_argument("-r", "--sourcePath", help="Path to the test datasets")
    parser.add_argument("-t", "--targetPath", help="Path to the model outputs")
    parser.add_argument("-p", "--patchSize", type=int, default=256, help="Set patch size")

    options = parser.parse_args(args)
    return options

def extractFileName(path, withoutExtension = None):
    ntpath.basename("a/b/c")
    head, tail = ntpath.split(path)

    if withoutExtension:
        return tail.split(".")[-2] or ntpath.basename(head).split(".")[-2]

    return tail or ntpath.basename(head)


def createDir(path):
    # Create a directory to save processed samples
    Path(path).mkdir(parents=True, exist_ok=True)
    return True

def imageList(path, multiDir = False, imageExtension =['*.jpg', '*.png', '*.jpeg', '*.tif', '*.npy']):
    #types = () # the tuple of file types
    imageList = []
    for ext in imageExtension:

        if multiDir == True:
            imageList.extend(glob.glob(path+"*/"+ext))
        else:
            imageList.extend(glob.glob(path+ext))
        
        imageList
    return imageList


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
    #print("Read Max: ", align_ratio)
    # Load image without changing bit depth and normalize by align ratio
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio

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
    #print("Write Max: ",align_ratio)
    np.save(alignratio_path, align_ratio)
    uint16_image_gt = np.round(image * align_ratio).astype(np.uint16)
    cv2.imwrite(image_path, cv2.cvtColor(uint16_image_gt, cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(uint16_image_gt, cv2.COLOR_RGB2BGR)






class HDMHDRProcess:
    def __init__(self, rootDir, targetDir, patchSize = 256 ):
        self.targetDir = targetDir
        self.rootDir = rootDir
        
        self.targetDir = targetDir
        createDir(self.targetDir)
        self.rootDir = rootDir

        #gamma=2.24

        self.patchSize = patchSize
    def patchExtract (self):
        images = imageList(self.rootDir, multiDir=True)
        print(len(images))
        countT = 0
        countGT = 0
        for im in images:
            if '_medium' in im:
                
                # Path Defination
                lognPath = im.replace("_medium", "_long")
                shortPath = im.replace("_medium", "_short")
                gtPath = im.replace("_medium", "_gt")
                alignPath = im.replace("_medium.png", '_alignratio.npy')
                exposurePath = im.replace("_medium.png", "_exposures.npy")

                # Read Images
                image_long = cv2.imread(lognPath, cv2.IMREAD_UNCHANGED) / 255.0
                image_short = cv2.imread(shortPath, cv2.IMREAD_UNCHANGED)  / 255.0
                image_medium = cv2.imread(im, cv2.IMREAD_UNCHANGED) / 255.0
                imageGT16 = imread_uint16_png(gtPath, alignPath)


                # Exposure Normalization 
                exposures=np.load(exposurePath)
                floating_exposures = exposures - exposures[1]

                image_short_corrected = image_short#(((image_short**gamma)*2.0**(-1*floating_exposures[0]))**(1/gamma))
                image_long_corrected = image_long#(((image_long**gamma)*2.0**(-1*floating_exposures[2]))**(1/gamma))
                image_medium_corrected = image_medium

                # Extracting Patch
                imgTemp = imageGT16[:imageGT16.shape[0]- imageGT16.shape[0] % patchSize, : imageGT16.shape[1]- imageGT16.shape[1] % patchSize]
                for i in range(0, imgTemp.shape[0],  patchSize):
                    for j in range(0, imgTemp.shape[1],  patchSize):
                        
                        image_long_corrected_crop = image_long_corrected[i:i+ patchSize, j:j+ patchSize, :]
                        image_short_corrected_crop = image_short_corrected[i:i+ patchSize, j:j+ patchSize, :]
                        image_medium_crop = image_medium_corrected[i:i+ patchSize, j:j+ patchSize, :]
                        imageGT_crop = imageGT16[i:i+ patchSize, j:j+ patchSize, :]
            
                        targetCounter = "{:05d}".format(countT) + "_"

                        cv2.imwrite(self.targetDir + targetCounter + "medium.png", image_medium_crop *255 )
                        cv2.imwrite(self.targetDir + targetCounter + "long.png", image_long_corrected_crop *255 )
                        cv2.imwrite(self.targetDir + targetCounter + "short.png", image_short_corrected_crop *255 )
                        cv2.imwrite(self.targetDir + targetCounter + "gt8.png", cv2.cvtColor(imageGT_crop, cv2.COLOR_RGB2BGR) *255 )
                        imwrite_uint16_png(self.targetDir + targetCounter + "gt16.png", imageGT_crop, self.targetDir + targetCounter + "alignratio.npy" )

                        countT += 1
                        if countT % 100 == 0:
                            print(countT)

                
   
    def __call__(self):
        self.patchExtract()
        #self.modelEvaluation()

if __name__ == "__main__":

    options = mainParser(sys.argv[1:])
    if len(sys.argv) == 1:
        customPrint("Invalid option(s) selected! To get help, execute script with -h flag.")
        exit()

    if options.sourcePath:
        sourcePath = options.sourcePath
    if options.targetPath:
        targetPath = options.targetPath

    if options.patchSize:
        patchSize = options.patchSize
    else:
        patchSize = ""
    
    
    hdmHDR = HDMHDRProcess(sourcePath, targetPath, patchSize)
    hdmHDR()