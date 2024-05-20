import torch
import torch.nn.functional as F
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_2(n):
    return (n ** 2).sum([1, 2, 3]) ** 0.5

class FGSM_Attack:
    def __init__(self, model, epsilon):
        self.model = model
        self.epsilon = epsilon

    def attack(self, inputs, labels):
        x = inputs.detach()
        x.requires_grad_()
        with torch.enable_grad():
            output = self.model(x)
            loss = criterion(output, labels)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + self.epsilon * torch.sign(grad.detach())
        x = torch.clamp(x, 0, 1)
        return x


class Grad_Align_Attack:
    def __init__(self, model, epsilon, alpha):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha

    def attack(self, inputs, labels):
        x1 = inputs.detach()
        delta = torch.zeros_like(x1).uniform_(-self.epsilon, self.epsilon)
        x2 = torch.clamp(x1 + delta, 0, 1)
        x1.requires_grad_()
        x2.requires_grad_()
        with torch.enable_grad():
            output1 = self.model(x1)
            output2 = self.model(x2)
            loss1 = criterion(output1, labels)
            loss2 = criterion(output2, labels)
        grad1 = torch.autograd.grad(loss1, [x1])[0]
        grad2 = torch.autograd.grad(loss2, [x2])[0]
        cos = (grad1 * grad2).sum([1, 2, 3]) / (norm_2(grad1) * norm_2(grad2))

        delta = delta + self.alpha * torch.sign(grad2.detach())
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        x_adv = torch.clamp(x2 + delta, 0, 1)

        return x_adv, cos


class Fast_at_Attack:
    def __init__(self, model, epsilon, alpha):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha

    def attack(self, inputs, labels):
        x = inputs.detach()
        delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)
        x_adv.requires_grad_()
        with torch.enable_grad():
            output = self.model(x_adv)
            loss = criterion(output, labels)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        delta = delta + self.alpha * torch.sign(grad.detach())
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv

class N_FGSM_Attack:
    def __init__(self, model, epsilon, alpha):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha

    def attack(self, inputs, labels):
        x = inputs.detach()
        delta = torch.zeros_like(x).uniform_(-self.epsilon * 2, self.epsilon * 2)
        x_adv = torch.clamp(x + delta, 0, 1)
        x_adv.requires_grad_()
        with torch.enable_grad():
            output = self.model(x_adv)
            loss = criterion(output, labels)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        delta = delta + self.alpha * torch.sign(grad.detach())
        x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv


class S_FGSM_Attack:
    def __init__(self, model, epsilon, alpha, c):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.c = c

    def attack(self, inputs, labels):
        x = inputs.detach()
        delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)
        x_adv.requires_grad_()
        with torch.enable_grad():
            output = self.model(x_adv)
            loss = criterion(output, labels)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        delta = delta + self.alpha * torch.sign(grad.detach())
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        X_adv = torch.zeros_like(inputs)

        change = torch.ones_like(labels)

        for j in range(1, self.c+1):
            with torch.no_grad():
                x_check = torch.clamp(x + delta * j / self.c, 0, 1)
                output_check = self.model(x_check)
                _, predict_check = torch.max(output_check.data, 1)
                wrong_check = torch.masked_select(torch.arange(len(labels)).to(device), predict_check != labels)
                no_change = torch.masked_select(torch.arange(len(labels)).to(device), change == 1)
                change_idx = torch.masked_select(wrong_check, torch.isin(wrong_check, no_change))
                X_adv[change_idx] = x_check[change_idx]
                change[change_idx] = 0

        correct_idx = torch.masked_select(torch.arange(len(labels)).to(device), change == 1)
        X_adv[correct_idx] = torch.clamp(x[correct_idx] + delta[correct_idx] / self.c, 0, 1)
        # X_adv = x + delta / self.c

        return X_adv