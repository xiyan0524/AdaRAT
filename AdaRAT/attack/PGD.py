import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn

criterion_kl = nn.KLDivLoss(reduction='batchmean')
criterion_ce = nn.CrossEntropyLoss()

EPS = 1e-8

class PGD_Attack:
    def __init__(self, model, epsilon, step_size, iteration):
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.iteration = iteration

    def attack_fosc(self, inputs, labels, min_fosc):
        x = inputs.detach() + torch.zeros_like(inputs).uniform_(-self.epsilon, self.epsilon)
        x = torch.clamp(x, 0, 1)
        control = torch.ones_like(inputs)
        for i in range(self.iteration):
            if i == 0:
                x.requires_grad_()
                with torch.enable_grad():
                    outputs = self.model(x)
                    loss = F.cross_entropy(outputs, labels, reduction='sum')
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + self.step_size * torch.sign(grad.detach())
            else:
                x = x.detach() + control * self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

            x.requires_grad_()
            with torch.enable_grad():
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, labels, reduction='sum')
            grad = torch.autograd.grad(loss, [x])[0]

            fosc = self.epsilon * torch.norm(grad.detach(), p=1, dim=(1, 2, 3)) - \
                   torch.sum(torch.mul(x - inputs, grad.detach()), dim=(1, 2, 3))
            idx = torch.masked_select(torch.arange(len(labels)).to(device), fosc <= min_fosc)
            control[idx] = 0

        # print(torch.mean(fosc))
        return x

    def attack_trades(self, inputs):
        x = inputs.detach() + torch.zeros_like(inputs).uniform_(-self.epsilon, self.epsilon)
        output_clean = self.model(inputs)
        x = torch.clamp(x, 0, 1)
        for i in range(self.iteration):
            x.requires_grad_()
            with torch.enable_grad():
                outputs_adv = self.model(x)
                loss = criterion_kl(F.log_softmax(outputs_adv, dim=1), F.softmax(output_clean, dim=1))
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x

    def attack_kl(self, inputs, label, class_num):
        x = inputs.detach() + torch.zeros_like(inputs).uniform_(-self.epsilon, self.epsilon)
        x = torch.clamp(x, 0, 1)
        for i in range(self.iteration):
            x.requires_grad_()
            with torch.enable_grad():
                outputs = self.model(x)
                loss = criterion_kl(F.log_softmax(outputs, dim=1), F.one_hot(label, class_num).to(torch.float32))
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x


    def A3T_attack(self, inputs, lables):
        X_adv = torch.zeros_like(inputs)
        with torch.no_grad():
            logits = self.model(inputs)
            predicted = torch.argmax(logits, dim=1)
        incorrect = torch.masked_select(torch.arange(len(lables)).to(device), predicted != lables)
        correct = torch.masked_select(torch.arange(len(lables)).to(device), predicted == lables)
        X_adv[incorrect] = inputs[incorrect]
        x = inputs.detach() + torch.zeros_like(inputs).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.iteration):
            x.requires_grad_()
            with torch.enable_grad():
                outputs = self.model(x)
                loss = criterion_ce(outputs, lables)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        X_adv[correct] = x[correct]
        return X_adv


    def attack(self, inputs, labels):
        # self.model.eval()
        x = inputs.detach() + torch.zeros_like(inputs).uniform_(-self.epsilon, self.epsilon)
        x = torch.clamp(x, 0, 1)
        for i in range(self.iteration):
            x.requires_grad_()
            with torch.enable_grad():
                outputs = self.model(x)
                loss = criterion_ce(outputs, labels)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x


class FAT_Attack:
    def __init__(self, model, epsilon, step_size, iteration, t):
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.iteration = iteration
        self.t = t

    def attack(self, inputs, labels):
        x_adv = torch.zeros_like(inputs).detach()
        chance = torch.ones_like(labels) * self.t
        flag = torch.ones_like(labels)
        x = inputs.detach() + torch.zeros_like(inputs).uniform_(-self.epsilon, self.epsilon)
        x = torch.clamp(x, 0, 1)
        for i in range(self.iteration):
            x.requires_grad_()
            with torch.enable_grad():
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                # correct = (predicted == labels)
                wrong_idx = torch.masked_select(torch.arange(len(labels)).to(device), predicted != labels)
                chance[wrong_idx] = chance[wrong_idx] - 1
                no_chance = torch.masked_select(torch.arange(len(labels)).to(device), chance == -1)
                no_change = torch.masked_select(torch.arange(len(labels)).to(device), flag == 1)
                break_idx = torch.masked_select(no_chance, torch.isin(no_chance, no_change))
                x_adv[break_idx] = x[break_idx]
                flag[break_idx] = 0

                loss = criterion_ce(outputs, labels)
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + self.step_size * torch.sign(grad.detach())
                x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
                x = torch.clamp(x, 0, 1)

        idx = torch.masked_select(torch.arange(len(labels)).to(device), flag != 0)
        x_adv[idx] = x[idx]

        return x_adv

class adaAD_Attack:
    def __init__(self, model, teacher, epsilon, step_size, iteration):
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.iteration = iteration
        self.teacher = teacher

    def attack(self, inputs):
        criterion = nn.KLDivLoss(reduction='none')
        self.model.eval()
        self.teacher.eval()
        x = inputs.detach() + 0.001 * torch.randn(inputs.shape).cuda().detach()
        for _ in range(self.iteration):
            x.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion(F.log_softmax(self.model(x), dim=1),
                                       F.softmax(self.teacher(x), dim=1))
                loss_kl = torch.sum(loss_kl)
            grad = torch.autograd.grad(loss_kl, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs-self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1.0)

        self.model.train()
        x = Variable(torch.clamp(x, 0, 1.0), requires_grad=False)

        return x
