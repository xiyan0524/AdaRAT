import sys
import os

import torchattacks
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

parser = argparse.ArgumentParser(description='TRADES')
parser.add_argument('--dataset_path', default='../../data', type=str, help='dataset path')
parser.add_argument('--epoch', default=120, type=int, help='train epoch')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--drop_last', default=False, type=bool, help='use drop last batch')
parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--class_num', default=10, type=int, help='class number')
parser.add_argument('--epsilon', default=8 / 255, type=float, help='max perturbed')
parser.add_argument('--step_size', default=2 / 255, type=float, help='step size')
parser.add_argument('--step_num', default=10, type=int, help='step numbers')
parser.add_argument('--lamda', default=6, type=int, help='lamda')

parser.add_argument('--FixRAT', default=False, type=bool, help='use new regularization')
parser.add_argument('--beta', default=0.03, type=float, help='beta')

parser.add_argument('--AdaRAT', default=False, type=bool, help='use two-stage optimization')
parser.add_argument('--up_lim', default=1.5, type=float, help='up limit')
parser.add_argument('--down_lim', default=0.7, type=float, help='down limit')
parser.add_argument('--tao', default=0.01, type=float, help='tao')

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
criterion_kl = nn.KLDivLoss(reduction="batchmean")

def FixRAT(outputs, perturbed_outputs, labels):
    y = F.one_hot(labels, args.class_num)
    new_regularization = args.beta / args.epsilon * (
            -y * torch.log_softmax(perturbed_outputs, dim=1) + y * torch.log_softmax(outputs, dim=1))
    new_regularization = torch.mean(torch.sum(new_regularization, dim=1))
    return new_regularization

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
    EPS = 1e-8
    method = "TRADES"
    if args.FixRAT:
        method += "_FixRAT"
        if args.AdaRAT:
            method += "_AdaRAT"
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
    title = "epoch   learning_rate   train_time   train_AE_loss   train_AE_acc   train_clean_loss   train_clean_acc " \
            "  test_clean_loss   test_clean_acc   test_FGSM_loss   test_FGSM_acc   test_PGD_loss   test_PGD_acc"
    logger.info(title)

    Clean = []
    Pgd = []

    for epoch in range(0, args.epoch):
        # 设置优化器
        optimizer = optim.SGD(net.parameters(), lr=adjust_Learning_Rate(epoch), momentum=0.9, weight_decay=5e-4)

        total = 0
        train_clean_loss = 0.0
        train_clean_acc = 0.0
        train_ae_loss = 0.0
        train_ae_acc = 0.0
        start_time = 0.0
        end_time = 0.0

        start_time = time.time()
        for i, data in enumerate(train_loader, 0):
            net.train()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            train_attack = PGD.PGD_Attack(net, args.epsilon, args.step_size, args.step_num)
            perturbed_inputs = train_attack.attack_trades(inputs)

            clean_outputs = net(inputs)
            perturbed_outputs = net(perturbed_inputs)

            clean_loss = criterion(clean_outputs, labels)
            perturbed_loss = criterion(perturbed_outputs, labels)
            trades_loss = criterion_kl(F.log_softmax(perturbed_outputs, dim=1), torch.clamp(F.softmax(clean_outputs, dim=1), min=EPS))
            reg = 0.0
            if args.FixRAT:
                if args.AdaRAT:
                    if clean_loss.item() >= args.up_lim * 0.7:
                        opt_loss = clean_loss
                    else:
                        opt_loss = (torch.pow(
                            torch.max(
                                torch.zeros_like(clean_loss),
                                torch.tan(
                                    math.pi * clean_loss / (2 * (args.up_lim - args.down_lim)) -
                                    math.pi * args.down_lim / (2 * (args.up_lim - args.down_lim))
                                )
                            ),
                            1.0 + args.tao)
                            + trades_loss * args.lamda
                            + FixRAT(clean_outputs, perturbed_outputs, labels))
                else:
                    opt_loss = clean_loss + trades_loss * args.lamda + FixRAT(clean_outputs, perturbed_outputs, labels)
            else:
                opt_loss = clean_loss + trades_loss * args.lamda

            opt_loss.backward()
            optimizer.step()

            total += labels.size(0)

            train_clean_loss += clean_loss.item()
            _, clean_predicted = torch.max(clean_outputs.data, 1)
            train_clean_acc += clean_predicted.eq(labels.data).cpu().sum()

            train_ae_loss += perturbed_loss.item()
            _, perturbed_predicted = torch.max(perturbed_outputs.data, 1)
            train_ae_acc += perturbed_predicted.eq(labels.data).cpu().sum()

        end_time = time.time()
        train_clean_loss = train_clean_loss / len(train_loader)
        train_clean_acc = train_clean_acc / total * 100.0

        train_ae_loss = train_ae_loss / len(train_loader)
        train_ae_acc = train_ae_acc / total * 100.0

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

        logger.info("%03d     %.4f          %06.2f       %07.4f         %05.2f%%         %07.4f            %05.2f%%"
                    "            %07.4f           %05.2f%%           %07.4f          %05.2f%%          %07.4f       "
                    "  %05.2f%%"
                    % (epoch + 1, adjust_Learning_Rate(epoch), end_time - start_time, train_ae_loss, train_ae_acc,
                       train_clean_loss, train_clean_acc, test_clean_loss, test_clean_acc, test_FGSM_loss,
                       test_FGSM_acc,
                       test_PGD_loss, test_PGD_acc)
                    )

    torch.save(net.state_dict(), "../../result/cifar10/save_model/" + method + ".pth")
    np.save("../../result/cifar10/data/" + method + "_clean.npy", Clean)
    np.save("../../result/cifar10/data/" + method + "_pgd.npy", Pgd)

    print("final robust evaluate:")

    PGD10_acc = 0.0
    PGD20_acc = 0.0
    PGD50_acc = 0.0
    FGSM_acc = 0.0
    CW_acc = 0.0
    AA_acc = 0.0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            FGSM_attack = FGSM.FGSM_Attack(net, args.epsilon)
            PGD10_attack = PGD.PGD_Attack(net, args.epsilon, args.step_size, iteration=10)
            PGD20_attack = PGD.PGD_Attack(net, args.epsilon, args.step_size, iteration=20)
            PGD50_attack = PGD.PGD_Attack(net, args.epsilon, args.step_size, iteration=50)
            # CW_attack = torchattacks.CW(net, c=0.01, kappa=0, steps=20, lr=0.1)
            CW_attack = cw.CW_Attack(net, args.epsilon, 20, args.step_size, 10)
            AA_attack = torchattacks.AutoAttack(net)

            with torch.enable_grad():
                PGD10_images = PGD10_attack.attack(images, labels)
                PGD20_images = PGD20_attack.attack(images, labels)
                PGD50_images = PGD50_attack.attack(images, labels)
                FGSM_images = FGSM_attack.attack(images, labels)
                CW_images = CW_attack.attack(images, labels)
                AA_images = AA_attack(images, labels)

            PGD10_outputs = net(PGD10_images)
            PGD20_outputs = net(PGD20_images)
            PGD50_outputs = net(PGD50_images)
            FGSM_outputs = net(FGSM_images)
            CW_outputs = net(CW_images)
            AA_outputs = net(AA_images)

            total += labels.size(0)

            _, PGD10_predicted = torch.max(PGD10_outputs.data, 1)
            PGD10_acc += (PGD10_predicted == labels).sum()
            _, PGD20_predicted = torch.max(PGD20_outputs.data, 1)
            PGD20_acc += (PGD20_predicted == labels).sum()
            _, PGD50_predicted = torch.max(PGD50_outputs.data, 1)
            PGD50_acc += (PGD50_predicted == labels).sum()
            _, FGSM_predicted = torch.max(FGSM_outputs.data, 1)
            FGSM_acc += (FGSM_predicted == labels).sum()
            _, CW_predicted = torch.max(CW_outputs.data, 1)
            CW_acc += (CW_predicted == labels).sum()
            _, AA_predicted = torch.max(AA_outputs.data, 1)
            AA_acc += (AA_predicted == labels).sum()

        PGD10_acc = PGD10_acc / total * 100.0
        PGD20_acc = PGD20_acc / total * 100.0
        PGD50_acc = PGD50_acc / total * 100.0
        FGSM_acc = FGSM_acc / total * 100.0
        CW_acc = CW_acc / total * 100.0
        AA_acc = AA_acc / total * 100.0

        logger.info(" ")
        logger.info(" ")
        logger.info("final robust result:")
        logger.info("FGSM ACC: %05.2f" % FGSM_acc)
        logger.info("PGD10 ACC: %05.2f" % PGD10_acc)
        logger.info("PGD20 ACC: %05.2f" % PGD20_acc)
        logger.info("PGD50 ACC: %05.2f" % PGD50_acc)
        logger.info("CW ACC: %05.2f" % CW_acc)
        logger.info("AA ACC: %05.2f" % AA_acc)

    print("train finish!")
