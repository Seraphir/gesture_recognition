import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import vgg16
from .C3D_model import C3D


class MergeNet(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()
        self.net2d = vgg16(pretrained=False, num_classes=num_classes)
        self.lc2d = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True)
        )
        self.net3d = C3D(pretrained=False, num_classes=num_classes)
        self.lc3d = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x1, x2):
        x1 = self.net2d.features(x1)
        x1 = self.net2d.avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.lc2d(x1)
        x2 = self.net3d.features(x2)
        x2 = x2.view(-1, 8192)
        x2 = self.lc3d(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        # F.log_softmax(x, dim=1)
        return x


# if __name__ == "__main__":
#     from torchviz import make_dot
#
#     x1 = torch.rand(1, 3, 224, 224)
#     x2 = torch.rand(1, 3, 32, 112, 112)
#     net = MergeNet(num_classes=3)
#     print(net)
#     outputs = net.forward(x1, x2)
#     g = make_dot(outputs)
#     g.render('espnet_model', view=False)
#     print(outputs.size())
