import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import sys
import glob
import time
import datetime
import colorama
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
from torchsummary import summary
from ptflops import get_model_complexity_info
from utilities.torchUtils import *
from dataTools.customDataloader import *
from utilities.inferenceUtils import *
from utilities.aestheticUtils import *
from loss.pytorch_msssim import *
from loss.colorLoss import *
from loss.percetualLoss import *
from modelDefinitions.attentionDis import *
from modelDefinitions.MKHRD import *
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.optim as optim

class DPBS:
    def __init__(self, config):
        
        # Model Configration 
        self.gtPath = config['gtPath']
        self.targetPath = config['targetPath']
        self.checkpointPath = config['checkpointPath']
        self.logPath = config['logPath']
        self.testImagesPath = config['testImagePath']
        self.resultDir = config['resultDir']
        self.modelName = config['modelName']
        self.dataSamples = config['dataSamples']
        self.batchSize = int(config['batchSize'])
        self.imageH = int(config['imageH'])
        self.imageW = int(config['imageW'])
        self.inputC = int(config['inputC'])
        self.outputC = int(config['outputC'])
        self.scalingFactor = int(config['scalingFactor'])
        self.binnigFactor = int(config['binnigFactor'])
        self.totalEpoch = int(config['epoch'])
        self.interval = int(config['interval'])
        self.learningRate = float(config['learningRate'])
        self.adamBeta1 = float(config['adamBeta1'])
        self.adamBeta2 = float(config['adamBeta2'])
        self.barLen = int(config['barLen'])
        
        # Initiating Training Parameters(for step)
        self.currentEpoch = 0
        self.startSteps = 0
        self.totalSteps = 0
        self.adversarialMean = 0

        # Normalization
        self.unNorm = UnNormalize()

        # Noise Level for inferencing
        self.noiseSet = [10, 20, 50]
        

        # Preapring model(s) for GPU acceleration
        self.device =  torch.device('cuda:0') #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#
        self.attentionNet = ResMKHDR().to(self.device)
        self.HDRRec = HDRRangeNet().to(self.device)
        self.discriminator = attentiomDiscriminator().to(self.device)

        # Optimizers
        self.optimizerEG = torch.optim.Adam(self.attentionNet.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        self.optimizerER = torch.optim.Adam(self.HDRRec.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        self.optimizerED = torch.optim.Adam(self.discriminator.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        
        # Scheduler for Super Convergance
        self.scheduleLR = None
        
    def customTrainLoader(self, overFitTest = False):
        
        targetImageList = imageList(self.targetPath)
        targetImageList = [k for k in targetImageList if '_medium' in k]
        print ("Trining Samples (Input):", self.targetPath, len(targetImageList))

        if overFitTest == True:
            targetImageList = targetImageList[-1:]
        if self.dataSamples:
            targetImageList = targetImageList[:self.dataSamples]

        datasetReader = customDatasetReader(   
                                                image_list=targetImageList, 
                                                imagePathGT=self.gtPath,
                                                height = self.imageH,
                                                width = self.imageW,
                                            )

        self.trainLoader = torch.utils.data.DataLoader( dataset=datasetReader,
                                                        batch_size=self.batchSize, 
                                                        shuffle=True,
                                                        #num_workers=8
                                                        )
        
        return self.trainLoader

    def modelTraining(self, resumeTraning=False, overFitTest=False, dataSamples = None):
        
        if dataSamples:
            self.dataSamples = dataSamples 

        # Losses
        featureLoss = regularizedFeatureLoss().to(self.device)
        reconstructionLoss = torch.nn.L1Loss().to(self.device)
        mseLoss = nn.MSELoss()
        ssimLoss = MSSSIM().to(self.device)
        colorLoss = deltaEColorLoss(normalize=True).to(self.device)
        adversarialLoss = nn.BCELoss().to(self.device)
        
        # Overfitting Testing
        if overFitTest == True:
            customPrint(Fore.RED + "Over Fitting Testing with an arbitary image!", self.barLen)
            trainingImageLoader = self.customTrainLoader(overFitTest=True)
            self.interval = 1
            self.totalEpoch = 100000
        else:  
            trainingImageLoader = self.customTrainLoader()


        # Resuming Training
        if resumeTraning == True:
            self.modelLoad(cpu=True)
            try:
                pass#self.modelLoad()

            except:
                #print()
                customPrint(Fore.RED + "Would you like to start training from sketch (default: Y): ", textWidth=self.barLen)
                userInput = input() or "Y"
                if not (userInput == "Y" or userInput == "y"):
                    exit()
        

        # Starting Training
        customPrint('Training is about to begin using:' + Fore.YELLOW + '[{}]'.format(self.device).upper(), textWidth=self.barLen)
        
        # Initiating steps
        #print("len of tranLoader:", len(trainingImageLoader))
        self.totalSteps =  int(len(trainingImageLoader)*self.totalEpoch)
        
        # Instantiating Super Convergance 
        self.scheduleLG = optim.lr_scheduler.StepLR(self.optimizerEG, step_size=1, gamma=0.01)#optim.lr_scheduler.OneCycleLR(optimizer=self.optimizerEG, max_lr=self.learningRate, total_steps=self.totalSteps)
        self.scheduleLR = optim.lr_scheduler.StepLR(self.optimizerER, step_size=1, gamma=0.01)
        torch.cuda.empty_cache()
        # Initiating progress bar 
        log10 = np.log(10)
        MAX_DIFF = 2 
        bar = ProgressBar(self.totalSteps, max_width=int(self.barLen/2))
        currentStep = self.startSteps
        while currentStep < self.totalSteps:
            #print(currentStep, self.startSteps)
            lossEG = 0
            for LRImagesLeft, HRGTImages, HRGTImages16 in trainingImageLoader:
                
                ##############################
                #### Initiating Variables ####
                ##############################

                # Time tracker
                iterTime = time.time()

                # Updating Steps
                currentStep += 1
                if currentStep > self.totalSteps:
                    self.savingWeights(currentStep)
                    customPrint(Fore.YELLOW + "Training Completed Successfully!", textWidth=self.barLen)
                    exit()

                # Images
                rawInputLeft = LRImagesLeft.to(self.device)
                #lumImage = lumImage.to(self.device)
                #DL = DL.to(self.device)
                #RL = RL.to(self.device)
                #print(DL.shape)
                highResReal = HRGTImages.to(self.device)
                highResReal16 = HRGTImages16.to(self.device)
                #print(rawInputLeft.shape, highResReal.shape)
                # GAN Variables
                onesConst = torch.ones(rawInputLeft.shape[0], 1).to(self.device)
                targetReal = (torch.rand(rawInputLeft.shape[0],1) * 0.5 + 0.7).to(self.device)
                targetFake = (torch.rand(rawInputLeft.shape[0],1) * 0.3).to(self.device)

                ##############################
                ####### Training Phase #######
                ##############################
    
                # Image Generation
                highResFake = self.attentionNet(rawInputLeft)
                highResHDR = self.HDRRec(highResFake.detach())
                
                if currentStep < 75000:
                    # Optimization of generator Stage I
                    self.optimizerEG.zero_grad()
                    generatorContentLoss =  reconstructionLoss(highResFake, highResReal) + (1 - ssimLoss(highResFake, highResReal)  )#+ colorLoss(highResFake, highResReal)
                    lossEG =  generatorContentLoss #+ 1e-3 * generatorAdversarialLoss 
                    lossEG.backward()
                    self.optimizerEG.step()

                
                # Optimaztion of Discriminator
                self.optimizerED.zero_grad()
                lossED = adversarialLoss(self.discriminator(highResReal16), targetReal) + \
                         adversarialLoss(self.discriminator(highResHDR.detach()), targetFake)
                lossED.backward()
                self.optimizerED.step()

                
                # Optimization of generator Stage II
                self.optimizerER.zero_grad()
                #edgeLossC = colorLoss(highResFake, highResReal)
                psnr = 10*torch.log( MAX_DIFF**2 / mseLoss(highResHDR, highResReal16) ) / log10
                generatorContentLoss =  reconstructionLoss(highResHDR, highResReal16) + colorLoss(highResHDR, highResReal16)

                generatorAdversarialLoss = adversarialLoss(self.discriminator(highResHDR), onesConst)
                lossER =  generatorContentLoss + 1e-3 * generatorAdversarialLoss #+ (1 - ssimLoss(highResFake, highResReal)  )
                lossER.backward()
                self.optimizerER.step()

                # Steps for Super Convergance
                if currentStep % 10000 == 0:
                    self.scheduleLG.step()
                    self.scheduleLR.step()

                ##########################
                ###### Model Logger ######
                ##########################   

                # Progress Bar
                if (currentStep  + 1) % 25 == 0:
                    bar.numerator = currentStep + 1
                    print(Fore.YELLOW + "Steps |",bar,Fore.YELLOW + "| LossEG: {:.4f}, LossER: {:.4f}, outMax: {:.4f}, gtMax: {:.4f}, PSNR: {:.4f}".format(lossEG, lossER, highResHDR.max(), highResReal16.max(), psnr),end='\r')
                    
                # Updating training log
                if (currentStep + 1) % 100 == 0:
                   
                    # Updating Tensorboard
                    summaryInfo = { 
                                    'Input Images' : rawInputLeft,
                                    'HDR Images (8Bit)' : highResFake,
                                    'HDR Images (16Bit)' : highResHDR,
                                    'GT Images (8Bit)' : highResReal,
                                    'GT Images (16Bit)' : highResReal16,
                                    'Step' : currentStep + 1,
                                    'Epoch' : self.currentEpoch,
                                    'LossEG' : lossEG.item(),
                                    'LossER' : lossER.item(),
                                    'LossED' : lossED.item(),
                                    'Path' : self.logPath,
                                    'Atttention Net' : self.attentionNet,
                                  }
                    tbLogWritter(summaryInfo)
                    save_image(self.unNorm(highResFake[0]), 'modelOutput.png')

                    # Saving Weights and state of the model for resume training 
                    self.savingWeights(currentStep)                
                
                if (currentStep + 1) % 10000 == 0 : 
                    # Epoch Summary
                    #print("\n")
                    eHours, eMinutes, eSeconds = timer(iterTime, time.time())
                    print (Fore.CYAN +'Steps [{}/{}] | Time elapsed [{:0>2}:{:0>2}:{:0>2}] | OutMax: {:.2f}, GTMax: {:.2f}, PSNR: {:.2f}, LossEG: {:.2f}, LossER: {:.2f}, LossED: {:.2f}' 
                            .format(currentStep + 1, self.totalSteps, eHours, eMinutes, eSeconds, highResFake.max() , highResReal.max(), psnr,lossEG, lossER, lossED))
                    self.savingWeights(currentStep + 1, True)
                    #self.modelInference(validation=True, steps = currentStep + 1)
            
            # Epoch Summary
            #eHours, eMinutes, eSeconds = timer(iterTime, time.time())
            #print (Fore.CYAN +'Steps [{}/{}] | Time elapsed [{:0>2}:{:0>2}:{:0>2}] | LossC: {:.2f}, LossP : {:.2f}, LossEG: {:.2f}, LossED: {:.2f}' 
            #        .format(currentStep + 1, self.totalSteps, eHours, eMinutes, eSeconds, colorLoss(highResFake, highResReal), featureLoss(highResFake, highResReal),lossEG, lossED))
            #self.savingWeights(currentStep, True)
            #self.modelInference(validation=True, steps = currentStep + 1)
            
   
    def modelInference(self, testImagesPath = None, outputDir = None, resize = None, validation = None, noiseSet = None, steps = None):
        if not validation:
            self.modelLoad(cpu=False)
            print("\nInferencing on pretrained weights.")
        else:
            print("Validation about to begin.")
        if not noiseSet:
            noiseSet = self.noiseSet
        if testImagesPath:
            self.testImagesPath = testImagesPath
        if outputDir:
            self.resultDir = outputDir
        

        modelInference = inference(gridSize=self.binnigFactor, inputRootDir=self.testImagesPath, outputRootDir=self.resultDir, modelName=self.modelName, validation=validation)

        testImageList = modelInference.testingSetProcessor()
        #print(testImageList, self.testImagesPath)
        barVal = ProgressBar(len(testImageList)/3, max_width=int(50))
        imageCounter = 0
        PSNRval = []
        SSIMVal = []
        c = 0
        from datetime import datetime
        with torch.no_grad():
            for imgPath in testImageList:
                torch.cuda.empty_cache()
                if "_medium" in imgPath:
                    #if int(extractFileName(imgPath, True).split("_")[0]) % 3 ==0:
                    #print(extractFileName(imgPath, True).split("_")[0])
                    c += 1
                    device = self.device
                    imgLDR, lumLDR = modelInference.inputForInference(imgPath, noiseLevel=0)#.to(self.device)
                    #print(imgL.shape, imgR.shape, imgPath)
                    a = datetime.now()
                    output = self.attentionNet(imgLDR.to(device))#.to(device)
                    
                    
                    torch.cuda.empty_cache()
                    output = self.HDRRec(output.detach())
                    b = datetime.now()
                    d = b - a
                    #print( d)
                    torch.cuda.empty_cache()
                    modelInference.saveModelOutput(output, imgPath, steps)
                     
                    imageCounter += 1
                    if imageCounter % 2 == 0:
                        barVal.numerator = imageCounter
                        print(Fore.CYAN + "Image Processd |", barVal,Fore.CYAN, end='\r')
            print(c)
            #print("\nSteps: {} | PSNR: {:.2f} | SSIM: {:.2f}".format(steps, np.mean(PSNRval), np.mean(SSIMVal)))
    
    def modelSummary(self,input_size = None):
        if not input_size:
            input_size = (3, self.imageH//self.scalingFactor, self.imageW//self.scalingFactor)

     
        customPrint(Fore.YELLOW + "AttentionNet (Generator)", textWidth=self.barLen)
        summary(self.attentionNet, input_size =input_size)
        print ("*" * self.barLen)
        print()

        customPrint(Fore.YELLOW + "AttentionNet (Discriminator)", textWidth=self.barLen)
        summary(self.discriminator, input_size =input_size)
        print ("*" * self.barLen)
        print()

        flops, params = get_model_complexity_info(self.attentionNet, input_size, as_strings=True, print_per_layer_stat=False)
        customPrint('Computational complexity (Enhace-Gen):{}'.format(flops), self.barLen, '-')
        customPrint('Number of parameters (Enhace-Gen):{}'.format(params), self.barLen, '-')

        flops, params = get_model_complexity_info(self.discriminator, input_size, as_strings=True, print_per_layer_stat=False)
        customPrint('Computational complexity (Enhace-Dis):{}'.format(flops), self.barLen, '-')
        customPrint('Number of parameters (Enhace-Dis):{}'.format(params), self.barLen, '-')
        print()

        configShower()
        print ("*" * self.barLen)
    
    def savingWeights(self, currentStep, duplicate=None):
        # Saving weights 
        checkpoint = { 
                        'step' : currentStep + 1,
                        'stateDictEG': self.attentionNet.state_dict(),
                        'stateDictER': self.HDRRec.state_dict(),
                        'stateDictED': self.discriminator.state_dict(),
                        'optimizerEG': self.optimizerEG.state_dict(),
                        'optimizerER': self.optimizerER.state_dict(),
                        'optimizerED': self.optimizerED.state_dict(),
                        'schedulerLR': self.scheduleLR
                        }
        saveCheckpoint(modelStates = checkpoint, path = self.checkpointPath, modelName = self.modelName)
        if duplicate:
            saveCheckpoint(modelStates = checkpoint, path = self.checkpointPath + str(currentStep) + "/", modelName = self.modelName, backup=None)

    def modelLoad(self, cpu=False):
        
        customPrint(Fore.RED + "Loading pretrained weight", textWidth=self.barLen)
        if cpu == True:
            previousWeight = loadCheckpoints(self.checkpointPath, self.modelName, True)
            #print(cpu)
        else:
            previousWeight = loadCheckpoints(self.checkpointPath, self.modelName)
        self.attentionNet.load_state_dict(previousWeight['stateDictEG'])
        self.HDRRec.load_state_dict(previousWeight['stateDictER'])
        self.discriminator.load_state_dict(previousWeight['stateDictED'])
        self.optimizerEG.load_state_dict(previousWeight['optimizerEG']) 
        self.optimizerER.load_state_dict(previousWeight['optimizerER']) 
        self.optimizerED.load_state_dict(previousWeight['optimizerED']) 
        self.scheduleLR = previousWeight['schedulerLR']
        self.startSteps = int(previousWeight['step'])
        #print(self.startSteps)
        
        customPrint(Fore.YELLOW + "Weight loaded successfully", textWidth=self.barLen)


        
