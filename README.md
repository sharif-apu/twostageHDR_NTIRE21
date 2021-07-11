# Two-stage LDR to HDR Image Reconstruction
This is the official implementation of paper title "A Two-stage Deep Network for High Dynamic Range Image Reconstruction". The paper has been accepted and expected to be published in the proceedings of CVPRW21. To download full paper **[[Click Here](https://arxiv.org/abs/2104.09386)]**.


**Please consider to cite this paper as follows:**
```
@inproceedings{a2021two,
  title={A two-stage deep network for high dynamic range image reconstruction},
  author={Sharif, SMA and Naqvi, Rizwan Ali and Biswas, Mithun and Kim, Sungjun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={550--559},
  year={2021}
}
```

# Overview
<p align="center">
<img width=800 align="center" src = "https://github.com/sharif-apu/twostageHDR_NTIRE21/blob/master/images/overviewUp.png" alt="Overview"> </br>
</p>

**Figure:** Overview of the proposed method.  The proposed method comprises a two-stage deep network.  Stage-I aims toperform image enhancement task denoising, exposure correction, etc.  Stage-II of the proposed method intends to performtone mapping and bit-expansion.


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

# LDR52 Dataset
We have collected a LDR dataset captured with different camera hardwares. Feel free to use our LDR dataset in your respective work. The dataset can be downloaded from the following link: **[[Click Here](https://drive.google.com/drive/u/1/folders/1vX4rM_953pAk83vNeWheiOiLzlnysZe9)]**
# Contact
For any further query, feel free to contact us through the following emails: sma.sharif.cse@ulab.edu.bd, rizwanali@sejong.ac.kr, or mithun.bishwash.cse@ulab.edu.bd

