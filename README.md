# Reconcile Accuracy and Robustness by Adaptive Ratio Adversarial Training
## Introduction
In this study, we investigated the phenomenon of the different synchronisation of Robust Accuracy (RA) and Standard Accuracy (SA) from the perspective of training loss (Figure 1). We found that regardless of the weight values chosen, there exists a trade-off between robustness and accuracy at a fixed ratio. Based on this, we have developed a new method called Adaptive Ratio Adversarial Training (AdaRAT), which utilises a novel radial unbounded barrier function to enhance the adversarial training loss (Figure 2), enabling the ratio between standard classification loss and the regularizer to adapt to the training progress. AdaRAT synchronises the peak performance of RA and SA, alleviating the trade-off between robustness and accuracy.

<div align=center>
<img src="AdaRAT/figure/fig1.png" width="500px"><img src="AdaRAT/figure/fig2.png" width="500px">
</div>

## Environment
* Python (3.9.18)
* Pytorch (2.1.2)
* torchvision (0.16.2)
* torchattacks(3.5.1)
* CUDA(12.2)
* Numpy(1.24.1)

## Folder contents
* ```AdaRAT/attack```: Code of adversarial attack (PGD et al.).
* ```AdaRAT/data```: Deposit of CIFAR-10 and CIFAR-100 datasets, and code for process the datasets.
* ```AdaRAT/model```: Code of model (Resnet-18 and WRN-34-10).
* ```AdaRAT/train```: Code of training (including PGD-AT, FAT, DAT, TRADES, KD-AT, IAD, Generalist and ST).
* ```AdaRAT/make_dir.py```: Create a folder for training results and trained models.
* ```AdaRAT/txt.py```: Write file and empty file operations.

## Training process
Create a folder for training results and models.
```bash
cd AdaRAT
python make_dir.py
```

Adversarial training (Take the training PGD-AT in CIFAR-10 dataset as an example).

* If you want to go to training with baseline, enter the following command:
```bash
cd AdaRAT/train/PGD-AT
python train_cifar10.py
```

* If you want to go to training with FixRAT, enter the following command:
```bash
cd AdaRAT/train/PGD-AT
python train_cifar10.py --FixRAT True
```

* If you want to go to training with AdaRAT, enter the following command:
```bash
cd AdaRAT/train/PGD-AT
python train_cifar10.py --FixRAT True --AdaRAT True
```

If you want to run KD-AT and IAD, make sure the file contains teacher models for both standard and adversarial training. The following code can be executed to create the teacher model:

```bash
cd AdaRAT/train/ST
python train_cifar10.py 
```

```bash
cd AdaRAT/train/PGD-AT
python train_cifar10.py 
```

You can run the following command to train the model under the CIFAR-100 dataset:
```bash
cd AdaRAT/train/PGD-AT
python train_cifar100.py --FixRAT True --AdaRAT True
```

After the model is trained, a final robustness test is performed, containing FGSM, PGD-10, PGD-20, PGD-50, cw and AutoAttack (AA).

## Reference Code
* PGD-AT: https://github.com/MadryLab/cifar10_challenge
* FAT: https://github.com/zjfheart/Friendly-Adversarial-Training
* DAT: https://github.com/YisenWang/dynamic_adv_training
* TRADRS: https://github.com/yaodongyu/TRADES/
* KD-AT: https://github.com/VITA-Group/Alleviate-Robust-Overfitting
* IAD: https://github.com/ZFancy/IAD
* Generalist: https://github.com/PKU-ML/Generalist
