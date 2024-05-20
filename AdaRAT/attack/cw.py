import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor


class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss


class CW_Attack:
    def __init__(self, model, epsilon, num_steps, step_size, class_num):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.model = model
        self.class_num = class_num

    def attack(self, X, y):
        X_pgd = Variable(X.data, requires_grad=True)
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-self.epsilon, self.epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(self.num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = CWLoss(self.class_num)(self.model(X_pgd), y)
            loss.backward()
            eta = self.step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -self.epsilon, self.epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        return X_pgd