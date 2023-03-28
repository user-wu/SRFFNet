# The Code for Paper "SRFFNet：Self-refine, Fusion, Feedback for Salient Object Detection".
by Shuang Wu, Guangjian Zhang
# Prerequisites
* Python 3.7
* Pytorch 1.7
* OpenCV 4.0
* Numpy 1.15
* TensorboardX
* [Apex](https://github.com/kezewang/apex)
# Clone repository
```
git clone git@github.com:user-wu/SRFFNet.git
cd SRFFNet/
```
# Download dataset 
Download the following datasets and unzip them into data folder
* PASCAL-S
* ECSSD
* HKU-IS
* DUT-OMRON
* DUTS

Directory Structure
```
 data --------------------------
      |-DUTS        -image/
      |             -mask/
      |             -test.txt
      |             -train.txt
      --------------------------
      |-DUt-OMRON   -image/
      |             -mask/
      |             -test.txt
      --------------------------
      |-ECSSD       -image/
      |             -mask/
      |             -test.txt
      --------------------------
      |-HKU-IS      -image/
      |             -mask/
      |             -test.txt
      --------------------------
      |-PASCAL-S    -image/
      |             -mask/
      |             -test.txt
      --------------------------
```
# Download model
* If you want to test the performance of SRFFNet, please download the [model](https://pan.baidu.com/s/1Yd55r7QuLkfe8qwCDMLkQw?pwd=rvji)  (extract code: ```rvji```) into out folder
* If you want to train your own model, please download the [pretrained model](https://download.pytorch.org/models/resnet50-19c8e357.pth) into ```res``` folder
# Training
```    
cd src/
python train.py
```
* ResNet-50 is used as the backbone of SRFFNet and DUTS-TR is used to train the model
* batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=32
* Warm-up and linear decay strategies are used to change the learning rate lr
* After training, the result models will be saved in out folder
# Testing
```
cd src
python3 test.py
```
* After testing, saliency maps of PASCAL-S, ECSSD, HKU-IS, DUT-OMRON, DUTS-TE will be saved in eval/maps/ folder.
* trained model: [model](https://pan.baidu.com/s/1Yd55r7QuLkfe8qwCDMLkQw?pwd=rvji)

# Citation
* If you find this work is helpful, please cite our paper
```
@article{wu2023srffnet,
  title={SRFFNet: Self-refine, Fusion and Feedback for Salient Object Detection},
  author={Wu, Shuang and Zhang, Guangjian},
  journal={Cognitive Computation},
  pages={1--13},
  year={2023},
  publisher={Springer}
}
```
