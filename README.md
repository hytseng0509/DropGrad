# Regularizing Meta-Learning via Gradient Dropout 
[[Paper]](https://arxiv.org/abs/2004.05859)

Pytorch implementation for our DropGrad approach. With the proposed regularization method, we can:

1. alleviate the overfitting problem in the exisiting gradient-based meta-learning models
2. improve the performance under **cross-domain** few-shot classification setting

Contact: Hung-Yu Tseng (htseng6@ucmerced.edu), Yi-Wen Chen (ychen319@ucmerced.edu)

## Paper
Please cite our paper if you find the code or dataset useful for your research.

Regularizing Meta-Learning via Gradient Dropout<br>
[Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/)\*, [Yi-Wen Chen](https://wenz116.github.io/)\*, [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/), [Sifei Liu](https://www.sifeiliu.net/), [Yen-Yu Lin](https://sites.google.com/site/yylinweb/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)<br>
ArXiv pre-print, 2020 (* equal contribution)
```
@article{dropgrad,
  author = {Tseng, Hung-Yu and Chen, Yi-Wen and Tsai, Yi-Hsuan and Liu, Sifei and Lin, Yen-Yu and Yang, Ming-Hsuan},
  title = {Regularizing Meta-Learning via Gradient Dropout},
  journal = {arXiv preprint arXiv:2004.05859},
  year = {2020}
}
```

## Usage

### Prerequisites
- Python >= 3.5
- Pytorch >= 1.3 and torchvision (https://pytorch.org/)
- You can use the `requirements.txt` file we provide to setup the environment via Anaconda.
```
conda create --name py36 python=3.6
conda install pytorch torchvision -c pytorch
pip3 install -r requirements.txt
```

### Install
Clone this repository:
```
git clone https://github.com/hytseng0509/DropGrad.git
cd DropGrad
```

### Datasets
Download 2 datasets seperately with the following commands.
- Set `DATASET_NAME` to: `cub`, `miniImagenet`.
```
cd filelists
python3 process.py DATASET_NAME
cd ..
```
- Refer to the instruction [here](https://github.com/wyharveychen/CloserLookFewShot#self-defined-setting) for constructing your own dataset.


### Training
Train gradient-based model on the mini-ImageNet dataset.
- `DPMETHOD` : dropout method `none`, `binary`, `gaussian`.
- `DPRATE`: dropout rate, we suggest 0.1.
```
python3 train.py --dropout_method DPMETHOD --dropout_rate DPRATE --name MAML_DPMETHOD_DPRATE --train_aug
```

### Evaluation
Test the model on the mini-ImageNet or CUB (cross-domain) dataset
- Specify `--dataset` to `miniImagenet` or `cub`
```
python3 test.py --name MAML_DPMETHOD_DPRATE --dataset TESTSET
```

## Note
- This code is built upon the implementation from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot).
- The dataset, model, and code are for non-commercial research purposes only.
- You can change the number of shot (i.e. 1/5 shots) using the argument `--n_shot`.
- Please refer to `output/checkpoints/download_models.py` for the example model file trained with the DropGrad approach.
