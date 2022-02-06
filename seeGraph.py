from torchviz import make_dot
import torch
import torchvision.models as models


x=torch.ones(10, requires_grad=True)
weights = {'x':x}

y=x**2
z=x**3
r=(y+z).sum()

make_dot(r).render("simple_graph", format="png")

#make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

resnet = models.resnet18(pretrained=True)

x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
out = resnet(x)
make_dot(out).render("resnet18", format="png")  # plot graph of variable, not of a nn.Module
make_dot(out, params=dict(list(resnet.named_parameters()))).render("resnet18a", format="png")  # plot graph of variable, not of a nn.Module

import hiddenlayer as hl

transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
graph = hl.build_graph(resnet, x, transforms=transforms)
graph.save('rnn_hiddenlayer', format='png')

graph1 = hl.build_graph(resnet, x)
graph1.theme = hl.graph.THEMES['blue'].copy()
graph1.save('rnn_hiddenlayer1', format='png')

torch.onnx.export(resnet, x, 'full_resnet.onnx')#, input_names=input_names, output_names=output_names)
torch.onnx.export(resnet, x, 'resnet.onnx', output_names=['Add_44'], input_names=['Add_6'])#, input_names=input_names, output_names=output_names)

f = '/home/borisef/projects/mmdet/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
m = torch.load(f)
