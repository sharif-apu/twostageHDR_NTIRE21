from utilities.parserUtils import *
from utilities.customUtils import *
from utilities.aestheticUtils import *
from dataTools.processDataset import *
from dataTools.patchExtractor import *
from mainModule.twostageHDR import *

if __name__ == "__main__":

    # Parsing Options
    options = mainParser(sys.argv[1:])
    if len(sys.argv) == 1:
        customPrint("Invalid option(s) selected! To get help, execute script with -h flag.")
        exit()
    
    # Reading Model Configuration
    if options.conf:
        configCreator()

    # Loading Configuration
    config = configReader()
  
    # Taking action as per received options
    if options.epoch:
        config=updateConfig(entity='epoch', value=options.epoch)
    if options.batch:
        config=updateConfig(entity='batchSize', value=options.batch)
    if options.manualUpdate:
        config=manualUpdateEntity()
    if options.modelSummary:
        twostageHDR(config).modelSummary()
    if options.train:
        twostageHDR(config).modelTraining(dataSamples=options.dataSamples)
    if options.retrain:
        twostageHDR(config).modelTraining(resumeTraning=True)
    if options.inference:
        twostageHDR(config).modelInference(options.sourceDir, options.resultDir)
    if options.overFitTest:
        twostageHDR(config).modelTraining(overFitTest=True)
 
        
            


