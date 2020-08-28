import torch
from model.model import VGG
from torchvision.models import AlexNet
from torchviz import make_dot
import tensorwatch as tw

x = torch.rand(1, 3, 224, 224)
model = AlexNet()
y = model(x)

# g = make_dot(y)
# g.render('espnet_model.pdf', view=False)

tw.draw_model(model, [1, 3, 224, 224])