import sys
import os

import torchattacks
from matplotlib import pyplot as plt

import logging
import random
import math
import numpy as np
import torch
from data import load_cifar10, load_cifar100
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from model import resnet, wide_resnet
import txt
from attack import FGSM, PGD, cw
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='ST')
parser.add_argument('--dataset_path', default='../../data', type=str, help='dataset path')
parser.add_argument('--epoch', default=120, type=int, help='train epoch')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--drop_last', default=False, type=bool, help='use drop last batch')
parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--class_num', default=10, type=int, help='class number')
parser.add_argument('--epsilon', default=8 / 255, type=float, help='max perturbed')
parser.add_argument('--step_size', default=2 / 255, type=float, help='step size')
parser.add_argument('--step_num', default=10, type=int, help='step numbers')

args = parser.parse_args()

train_loader, test_loader = load_cifar10.load(args.dataset_path, args.batch_size, args.drop_last)

net = resnet.ResNet18(args.class_num).to(device)

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

criterion = nn.CrossEntropyLoss()

def adjust_Learning_Rate(epoch):
    lr = args.learning_rate
    if epoch >= 60:
        lr = args.learning_rate * 0.1
    if epoch >= 90:
        lr = args.learning_rate * 0.01
    if epoch >= 110:
        lr = args.learning_rate * 0.005
    return lr


if __name__ == "__main__":
    method = "ST"
    print("start training!")
    print("train method: %s" % method)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join("../../result/cifar10/txt/", method + ".log")),
            logging.StreamHandler()
        ])

    txt.clear("../../result/cifar10/txt/" + method + ".log")

    logger.info("training method: %s" % method)
    logger.info(args)
    title = "epoch   learning_rate   train_time   train_clean_loss   train_clean_acc " \
            "  test_clean_loss   test_clean_acc   test_FGSM_loss   test_FGSM_acc   test_PGD_loss   test_PGD_acc"
    logger.info(title)

    Clean = []
    Pgd = []

    for epoch in range(0, args.epoch):
        optimizer = optim.SGD(net.parameters(), lr=adjust_Learning_Rate(epoch), momentum=0.9, weight_decay=5e-4)

        total = 0
        train_clean_loss = 0.0
        train_clean_acc = 0.0
        start_time = 0.0
        end_time = 0.0

        start_time = time.time()
        for i, data in enumerate(train_loader, 0):
            net.train()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            clean_outputs = net(inputs)

            clean_loss = criterion(clean_outputs, labels)

            opt_loss = clean_loss

            opt_loss.backward()
            optimizer.step()

            total += labels.size(0)

            train_clean_loss += clean_loss.item()
            _, clean_predicted = torch.max(clean_outputs.data, 1)
            train_clean_acc += clean_predicted.eq(labels.data).cpu().sum()

        end_time = time.time()
        train_clean_loss = train_clean_loss / len(train_loader)
        train_clean_acc = train_clean_acc / total * 100.0

        total = 0
        test_clean_loss = 0.0
        test_clean_acc = 0.0
        test_FGSM_loss = 0.0
        test_FGSM_acc = 0.0
        test_PGD_loss = 0.0
        test_PGD_acc = 0.0

        with torch.no_grad():
            for data in test_loader:
                net.eval()

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                test_FGSM_attack = FGSM.FGSM_Attack(net, args.epsilon)
                FGSM_inputs = test_FGSM_attack.attack(inputs, labels)

                test_PGD_attack = PGD.PGD_Attack(net, args.epsilon, args.step_size, args.step_num)
                PGD_inputs = test_PGD_attack.attack(inputs, labels)

                clean_outputs = net(inputs)
                FGSM_outputs = net(FGSM_inputs)
                PGD_outputs = net(PGD_inputs)

                clean_loss = criterion(clean_outputs, labels)
                FGSM_loss = criterion(FGSM_outputs, labels)
                PGD_loss = criterion(PGD_outputs, labels)

                total += labels.size(0)

                test_clean_loss += clean_loss.item()
                _, clean_predicted = torch.max(clean_outputs.data, 1)
                test_clean_acc += clean_predicted.eq(labels.data).cpu().sum()

                test_FGSM_loss += FGSM_loss.item()
                _, FGSM_predicted = torch.max(FGSM_outputs.data, 1)
                test_FGSM_acc += FGSM_predicted.eq(labels.data).cpu().sum()

                test_PGD_loss += PGD_loss.item()
                _, PGD_predicted = torch.max(PGD_outputs.data, 1)
                test_PGD_acc += PGD_predicted.eq(labels.data).cpu().sum()

            test_clean_loss = test_clean_loss / len(test_loader)
            test_clean_acc = test_clean_acc / total * 100.0
            Clean.append(test_clean_acc)

            test_FGSM_loss = test_FGSM_loss / len(test_loader)
            test_FGSM_acc = test_FGSM_acc / total * 100.0

            test_PGD_loss = test_PGD_loss / len(test_loader)
            test_PGD_acc = test_PGD_acc / total * 100.0
            Pgd.append(test_PGD_acc)

        logger.info("%03d     %.4f          %06.2f       %07.4f            %05.2f%%"
                    "            %07.4f           %05.2f%%           %07.4f          %05.2f%%          %07.4f       "
                    "  %05.2f%%"
                    % (epoch + 1, adjust_Learning_Rate(epoch), end_time - start_time,
                       train_clean_loss, train_clean_acc, test_clean_loss, test_clean_acc, test_FGSM_loss,
                       test_FGSM_acc,
                       test_PGD_loss, test_PGD_acc)
                    )

    torch.save(net.state_dict(), "../../result/cifar10/save_model/" + method + ".pth")
    np.save("../../result/cifar10/data/" + method + "_clean.npy", Clean)
    np.save("../../result/cifar10/data/" + method + "_pgd.npy", Pgd)

    print("train finish!")
