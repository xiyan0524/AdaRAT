import os
import argparse
import time

import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
from tqdm import tqdm
import numpy as np
import copy

from utils import Logger, AverageMeter
from attacks import *

parser = argparse.ArgumentParser(description='Generalist')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--arch', type=str, default="resnet18", help="decide which network to use,choose from smallcnn, resnet18, WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="dataset")
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')

parser.add_argument('--loss_fn', type=str, default="cent", help="loss function")
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step-size', type=float, default=0.007, help='step size')
parser.add_argument('--norm', type=str, default='Linf', help='type of attack')

parser.add_argument('--resume',type=bool, default=False, help='whether to resume training')
parser.add_argument('--out-dir',type=str, default='./logs/txt/10',help='dir of output')
parser.add_argument('--data-dir',type=str, default='./logs/data/10',help='dir of output')
parser.add_argument('--model-dir',type=str, default='./logs/model/10',help='dir of output')
parser.add_argument('--ablation', type=str, default='', help='ablation study')

parser.add_argument('--FixRAT', default=False, type=bool, help='use new regularization')
parser.add_argument('--beta', default=0.04, type=float, help='beta')

parser.add_argument('--AdaRAT', default=False, type=bool, help='use two-stage optimization')
parser.add_argument('--up_lim', default=1.3, type=float, help='up limit')
parser.add_argument('--down_lim', default=0.5, type=float, help='down limit')
parser.add_argument('--tao', default=0.01, type=float, help='tao')


args = parser.parse_args()
Clean = []
Pgd = []
# Training settings
args.out_dir = os.path.join(args.out_dir, args.ablation)
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

args.num_classes = 10 if args.dataset in ['cifar10', 'mnist', 'svhn'] else 100
weight_decay = 5e-4
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def FixRAT(outputs, perturbed_outputs, labels):
    y = F.one_hot(labels, num_classes=args.num_classes)
    new_regularization = args.beta / args.epsilon * (
            -y * torch.log_softmax(perturbed_outputs, dim=1) + y * torch.log_softmax(outputs, dim=1))
    new_regularization = torch.mean(torch.sum(new_regularization, dim=1))
    return new_regularization

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model, model2=None, beta=None):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        for i, name in enumerate(self.param_keys):
            if model2 and beta:
                if i < len(self.param_keys):
                    state2 = model2.state_dict()
                    self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * (state[name] * beta + state2[name] * (1-beta)))
            else:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for i, name in enumerate(self.buffer_keys):
            if self.buffer_ema:
                if model2 and beta:
                    if i < len(self.buffer_keys):
                        state2 = model2.state_dict()
                        self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * (state[name] * beta + state2[name] * (1-beta)))
                else:
                    self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }


if args.arch == 'resnet18':
    adjust_learning_rate = lambda epoch: np.interp([epoch], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs],
                                                   [args.lr, args.lr, args.lr / 10, args.lr / 100])[0]
elif args.arch == 'WRN':
    args.lr = 0.1
    adjust_learning_rate = lambda epoch: np.interp([epoch], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs],
                                                   [args.lr, args.lr, args.lr / 10, args.lr / 20])[0]

adjust_beta = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [1.0, 1.0, 1.0, 0.9])[
    0]


def train(epoch, model, model_st, teacher_at, teacher_st, teacher_mixed, Attackers, optimizer_ST, optimizer_AT, device,
          descrip_str):
    beta = 0.04
    teacher_at.model.eval()
    teacher_st.model.eval()
    teacher_mixed.model.eval()

    losses = AverageMeter()
    clean_accuracy = AverageMeter()
    adv_accuracy = AverageMeter()

    pbar = tqdm(train_loader)
    pbar.set_description(descrip_str)

    for batch_idx, (inputs, target) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, target = inputs.to(device), target.to(device)

        # loss, logit = trades_loss(model, inputs, target, optimizer, epoch, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0, distance='l_inf', device='device')
        # loss, logit = mart_loss(model, inputs, target, optimizer, epoch, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0, distance='l_inf', device='device')
        x_adv= Attackers.run_specified('PGD_10', model, inputs, target, return_acc=False)

        # For AT update
        model.train()
        lr = adjust_learning_rate(epoch)
        optimizer_AT.param_groups[0].update(lr=lr)
        optimizer_AT.zero_grad()
        clean_logit = model(inputs)
        adv_logit = model(x_adv)
        clean_loss = nn.CrossEntropyLoss()(clean_logit, target)
        adv_loss = nn.CrossEntropyLoss()(adv_logit, target)

        if args.FixRAT:
            if args.AdaRAT:
                if clean_loss.item() <= args.up_lim * 0.7:
                    loss_at = (torch.pow(
                                torch.max(
                                    torch.zeros_like(clean_loss),
                                    torch.tan(
                                        math.pi * clean_loss / (2 * (args.up_lim - args.down_lim)) -
                                        math.pi * args.down_lim / (2 * (args.up_lim - args.down_lim))
                                    )
                                ),
                                1.0 + args.tao)
                                + FixRAT(clean_logit, adv_logit, target))
                else:
                    loss_at = clean_loss
            else:
                loss_at = clean_loss + FixRAT(clean_logit, adv_logit, target)
        else:
            loss_at = adv_loss

        loss_at.backward()
        optimizer_AT.step()

        teacher_at.update_params(model)
        teacher_at.apply_shadow()

        # For ST update
        model_st.train()
        optimizer_ST.param_groups[0].update(lr=lr)
        optimizer_ST.zero_grad()
        nat_logit = model_st(inputs)
        loss_st = nn.CrossEntropyLoss()(nat_logit, target)
        loss_st.backward()
        optimizer_ST.step()

        teacher_st.update_params(model_st)
        teacher_st.apply_shadow()

        beta = adjust_beta(epoch)

        teacher_mixed.update_params(teacher_at.model, teacher_st.model, beta=beta)
        teacher_mixed.apply_shadow()

        if epoch >= 75 and epoch % 3 == 0:
            model.load_state_dict(teacher_mixed.shadow)
            model_st.load_state_dict(teacher_mixed.shadow)

        losses.update(loss_at.item())
        clean_accuracy.update(torch_accuracy(teacher_st.model(inputs), target, (1,))[0].item())
        adv_accuracy.update(torch_accuracy(teacher_at.model(inputs), target, (1,))[0].item())

        pbar_dic['loss'] = '{:.2f}'.format(losses.mean)
        pbar_dic['Acc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['advAcc'] = '{:.2f}'.format(adv_accuracy.mean)
        pbar.set_postfix(pbar_dic)


def test(model, model_st, Attackers, device):
    model.eval()
    model_st.eval()

    clean_accuracy = AverageMeter()
    adv_accuracy = AverageMeter()
    ema_clean_accuracy = AverageMeter()
    ema_adv_accuracy = AverageMeter()

    pbar = tqdm(test_loader)
    pbar.set_description('Testing')

    for batch_idx, (inputs, target) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, target = inputs.to(device), target.to(device)

        acc = Attackers.run_specified('NAT', model, inputs, target, return_acc=True)
        adv_acc = Attackers.run_specified('PGD_10', model, inputs, target, category='Madry', return_acc=True)

        ema_acc = Attackers.run_specified('NAT', model_st, inputs, target, return_acc=True)
        ema_adv_acc = Attackers.run_specified('PGD_10', model_st, inputs, target, category='Madry', return_acc=True)

        clean_accuracy.update(acc[0].item())
        adv_accuracy.update(adv_acc[0].item())
        ema_clean_accuracy.update(ema_acc[0].item())
        ema_adv_accuracy.update(ema_adv_acc[0].item())

        pbar_dic['cleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['advAcc'] = '{:.2f}'.format(adv_accuracy.mean)
        pbar_dic['ema_cleanAcc'] = '{:.2f}'.format(ema_clean_accuracy.mean)
        pbar_dic['ema_advAcc'] = '{:.2f}'.format(ema_adv_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    Clean.append(ema_clean_accuracy.mean)
    Pgd.append(ema_adv_accuracy.mean)

    return clean_accuracy.mean, adv_accuracy.mean, ema_clean_accuracy.mean, ema_adv_accuracy.mean



def attack(model, Attackers, device):
    model.eval()

    clean_accuracy = AverageMeter()
    fgsm_accuracy = AverageMeter()
    pgd10_accuracy = AverageMeter()
    pgd20_accuracy = AverageMeter()
    pgd50_accuracy = AverageMeter()
    # pgd100_accuracy = AverageMeter()
    # mim_accuracy = AverageMeter()
    cw_accuracy = AverageMeter()
    # APGD_ce_accuracy = AverageMeter()
    # APGD_dlr_accuracy = AverageMeter()
    # APGD_t_accuracy = AverageMeter()
    # FAB_t_accuracy = AverageMeter()
    # Square_accuracy = AverageMeter()
    aa_accuracy = AverageMeter()

    pbar = tqdm(test_loader)
    pbar.set_description('Attacking all')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, targets = inputs.to(device), targets.to(device)

        acc_dict = Attackers.run_all(model, inputs, targets)

        clean_accuracy.update(acc_dict['NAT'][0].item())
        fgsm_accuracy.update(acc_dict['FGSM'][0].item())
        pgd10_accuracy.update(acc_dict['PGD_10'][0].item())
        pgd20_accuracy.update(acc_dict['PGD_20'][0].item())
        pgd50_accuracy.update(acc_dict['PGD_50'][0].item())
        # pgd100_accuracy.update(acc_dict['PGD_100'][0].item())
        # mim_accuracy.update(acc_dict['MIM'][0].item())
        cw_accuracy.update(acc_dict['CW'][0].item())
        # APGD_ce_accuracy.update(acc_dict['APGD_ce'][0].item())
        # APGD_dlr_accuracy.update(acc_dict['APGD_dlr'][0].item())
        # APGD_t_accuracy.update(acc_dict['APGD_t'][0].item())
        # FAB_t_accuracy.update(acc_dict['FAB_t'][0].item())
        # Square_accuracy.update(acc_dict['Square'][0].item())
        aa_accuracy.update(acc_dict['AA'][0].item())

        pbar_dic['clean'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['FGSM'] = '{:.2f}'.format(fgsm_accuracy.mean)
        pbar_dic['PGD10'] = '{:.2f}'.format(pgd10_accuracy.mean)
        pbar_dic['PGD20'] = '{:.2f}'.format(pgd20_accuracy.mean)
        pbar_dic['PGD50'] = '{:.2f}'.format(pgd50_accuracy.mean)
        # pbar_dic['PGD100'] = '{:.2f}'.format(pgd100_accuracy.mean)
        # pbar_dic['MIM'] = '{:.2f}'.format(mim_accuracy.mean)
        pbar_dic['CW'] = '{:.2f}'.format(cw_accuracy.mean)
        # pbar_dic['APGD_ce'] = '{:.2f}'.format(APGD_ce_accuracy.mean)
        # pbar_dic['APGD_dlr'] = '{:.2f}'.format(APGD_dlr_accuracy.mean)
        # pbar_dic['APGD_t'] = '{:.2f}'.format(APGD_t_accuracy.mean)
        # pbar_dic['FAB_t'] = '{:.2f}'.format(FAB_t_accuracy.mean)
        # pbar_dic['Square'] = '{:.2f}'.format(Square_accuracy.mean)
        pbar_dic['AA'] = '{:.2f}'.format(aa_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    return [clean_accuracy.mean, fgsm_accuracy.mean, pgd10_accuracy.mean, pgd20_accuracy.mean, pgd50_accuracy.mean, cw_accuracy.mean, aa_accuracy.mean]


def main():
    # device = args.device
    device = torch.device(args.device)
    best_ema_acc_adv = 0
    start_epoch = 1
    TIME = 0.0


    if args.arch == "smallcnn":
        model = SmallCNN().to(device)
    if args.arch == "resnet18":
        model = ResNet18(num_classes=args.num_classes).to(device)
    if args.arch == "preactresnet18":
        model = PreActResNet18().to(device)
    if args.arch == "WRN32":
        model = Wide_ResNet_Madry(depth=32, num_classes=args.num_classes, widen_factor=10, dropRate=0.0).to(device)
    if args.arch == "WRN34":
        model = Wide_ResNet_Madry(depth=34, num_classes=args.num_classes, widen_factor=10, dropRate=0.0).to(device)

    method = "generalist"
    if args.FixRAT:
        method += "_FixRAT"
    elif args.AdaRAT:
        method += "_AdaRAT"

    # model_at = torch.nn.DataParallel(model, device_ids=[6, 7])
    model_at = model.to(device)
    model_st = copy.deepcopy(model_at)
    teacher_at = EMA(model_at)
    teacher_st = EMA(model_st)
    teacher_mixed = EMA(model_st)
    # model_at = model_at.to(device)
    Attackers = AttackerPolymer(args.epsilon, args.num_steps, args.step_size, args.num_classes, device)

    logger_test = Logger(os.path.join(args.out_dir,method + '.txt'), title='reweight')

    if not args.resume:
        optimizer_ST = optim.SGD(model_st.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        optimizer_AT = optim.SGD(model_at.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)

        logger_test.set_names(['Epoch', 'Natural', 'PGD20', 'ema_Natural', 'ema_PGD20', 'Time'])

        for epoch in range(start_epoch, args.epochs+1):

            descrip_str = 'Training epoch:{}/{}'.format(epoch, args.epochs)
            # start_time = time.time()
            start_time = 0.0
            end_time = 0.0
            start_time = time.time()
            train(epoch, model_at, model_st, teacher_at, teacher_st, teacher_mixed, Attackers, optimizer_ST, optimizer_AT, device, descrip_str)
            end_time = time.time()
            TIME += end_time - start_time
            # elapsed = round(time.time() - start_time)
            # elapsed = str(datetime.timedelta(seconds=elapsed))
            # print(elapsed)

            nat_acc, pgd20_acc, ema_nat_acc, ema_pgd20_acc = test(teacher_st.model, teacher_mixed.model, Attackers, device=device)

            logger_test.append([epoch, nat_acc, pgd20_acc, ema_nat_acc, ema_pgd20_acc, TIME])


        # Save the last checkpoint
        np.save(os.path.join(args.data_dir, method + '_clean.npy'), Clean)
        np.save(os.path.join(args.data_dir, method + '_pgd.npy'), Pgd)
        torch.save(model_at.state_dict(), os.path.join(args.model_dir, method + '.pth.tar'))

    teacher_mixed.model.load_state_dict(torch.load(os.path.join(args.model_dir, method + '.pth.tar')))
    res_list2 = attack(teacher_mixed.model, Attackers, device)

    logger_test.set_names(['Epoch', 'clean', 'FGSM', 'PGD10', 'PGD20', 'PGD50', 'CW', 'AA'])
    # logger_test.append([1000000, res_list1[0], res_list1[1], res_list1[2], res_list1[3], res_list1[4], res_list1[5], res_list1[6], res_list1[7], res_list1[8], res_list1[9], res_list1[10]])
    logger_test.append([1000001, res_list2[0], res_list2[1], res_list2[2], res_list2[3], res_list2[4], res_list2[5], res_list2[6]])

    logger_test.close()


if __name__ == '__main__':
    main()

