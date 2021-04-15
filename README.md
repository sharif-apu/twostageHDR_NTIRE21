# Two-satge LDR to HDR Image Reconstruction

# Overview
<p align="center">
<img width=800 align="center" src = "https://github.com/sharif-apu/twostageHDR_NTIRE21/blob/master/images/overviewUp.png" alt="Overview"> </br>
</p>

**Figure:** The overview of proposed network architecture. The proposed network incorporates novel dynamic residual attention blocks, which utilizes dynamic convolution and a noise gate. Also, the network leverage the residual learning along with the learning feature correlation.


# Comparison with state-of-the-art sigle-shot LDR to HDR Deep methods </br>

<p align="center">
<img width=800 align="center" src = "https://github.com/sharif-apu/twostageHDR_NTIRE21/blob/master/images/comp.png" alt="Overview"> </br>
</p>

**Figure:** Quantitative comparison between proposed method and existing learning-based single-shot LDR to HDR methods..

# Prerequisites
```
Python 3.8
CUDA 10.1 + CuDNN
pip
Virtual environment (optional)
```

# Installation
**Please consider using a virtual environment to continue the installation process.**
```
git clone https://github.com/sharif-apu/twostageHDR_NTIRE21.git
cd twostageHDR_NTIRE21
pip install -r requirement.txt
```
# Training
To download the training images please visit the following link: **[[Click Here](https://competitions.codalab.org/competitions/28161#participate)]** and extract the zip files in common directory.</br> 
The original paper used image patches from HdM HDR dataset. To extract image patches please execute Extras/processHDMDHR.py script from the root directory as follows:

```python processHDMDHR.py -r path/to/HdM/root/ -t path/to/save/patch -p 256```
</br> Here **-r** flag defines your root directory of the HdM HDR training samples, **-s** flag defines the directory where patches should be saved, and **-p** flag defines the patch size</br>

</br> After extracting patch please execute the following commands to start training:

```python main.py -ts -e X -b Y```
To specify your trining images path, go to mainModule/config.json and update "trainingImagePath" entity. </br>You can specify the number of epoch with **-e** flag (i.e., -e 5) and number of images per batch with **-b** flag (i.e., -b 24).</br>

*Please Note: The provided code aims to train only with medium exposure frames. To train with short/long exposure frames, you need to modify the dataTools/customDataloader (line 68) and mainModule/twostageHDR (line 87).*

**For transfer learning execute:**</br>
```python main.py -tr -e -b ```

# Testing
The provided weights are trained as per rule of NTIRE21 HDR challange (single frame). To download the testing images please visit the following link: **[[Click Here](https://competitions.codalab.org/competitions/28161#participate)]**

**To inference with provided pretrained weights please execute the following commands:**</br>
```python main.py -i -s path/to/inputImages -d path/to/outputImages ``` </br>
Here,**-s** specifies the root directory of the source images
 (i.e., testingImages/), and **-d** specifies the destination root (i.e., modelOutput/).


# Contact
For any further query, feel free to contact us through the following emails: sma.sharif.cse@ulab.edu.bd, rizwanali@sejong.ac.kr, or mithun.bishwash.cse@ulab.edu.bd
# LDR2HDR_CVPR21
