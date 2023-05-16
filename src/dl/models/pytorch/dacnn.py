# https://dejanbatanjac.github.io/resnet18
import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DACNN(nn.Module):

    def __init__(self, ni, no, ndom, dropout=0.0):
        super().__init__()
        self.l0 = nn.Conv2d(ni, 32, kernel_size=(7, 7), stride=(2, 2), padding=0)
        self.l1 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=0)
        self.l2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=0)
        self.l3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=0)

        self.l4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.l5 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.l6 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.l7 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.linear_classification = nn.Linear(512, no)
        self.linear_domain = nn.Linear(512, ndom)

        self.dropout = nn.Dropout2d(p=dropout)

        self.random_init()

    def forward(self, input, alpha=None):
        x = self.l0(input)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x).squeeze()

        # x = self.l4(x)
        # x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        # x = self.l8(x)
        # x = self.l9(x)
        # x = self.dropout(x)
        classif = self.linear_classification(x)
        dom = 0
        if alpha is not None:
            dom = self.linear_domain(
                ReverseLayerF.apply(x, alpha)
            )
        return classif, dom

    def inception(self, input):
        x = self.head(input)
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x).squeeze()
        # x = self.l6(x)
        # x = self.l7(x)
        # x = self.l8(x)
        # x = self.l9(x)

        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)

