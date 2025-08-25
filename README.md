# DcSplat
Official implementation of "DcSplat: Dual-Constraint Human Gaussian Splatting with Latent Multi-View Consistency".

## :fire: News
* **[2023.3.4]** We have created a code repository on [github](https://github.com/Xiaofei-CN/DPAGen) and will continue to update it in the future!
* **[2025.2.26]** Our paper [Disentangled Pose and Appearance Guidance for Multi-Pose Generation]() has been accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2025!

## Method
<img src=figure/overview.png>

## Installation

To deploy and run DPAGen, run the following scripts:
```
conda create -n dpagen python=3.10 -y
conda activate dpagen

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

# Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{xiao2025disentangled,
  title={Disentangled Pose and Appearance Guidance for Multi-Pose Generation},
  author={Xiao, Tengfei and Wu, Yue and Li, Yuelong and Qin, Can and Gong, Maoguo and Miao, Qiguang and Ma, Wenping},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={5646--5655},
  year={2025}
}
```
