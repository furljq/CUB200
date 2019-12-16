import numpy as np
import torch
import torch.nn.functional as F
from model import GCN
from graph import build_graph_skeleton as build_graph
from dataLoader import cropCUB

G = build_graph()

net = GCN(200, 500, 200)
net = net.cuda(0)

start_epoch = 0
if False:
    ckpt = torch.load('./ckpt/model_0.ckpt')
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1

datas = cropCUB()
dataLoader = torch.utils.data.DataLoader(datas, batch_size=8, shuffle=True, num_workers=4)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

epoch_losses = []
for epoch in range(start_epoch, 80):
    epoch_loss = 0
    for iter, (inputs, labels) in enumerate(dataLoader):
        inputs = inputs.cuda(0)
        labels = labels.cuda(0)
        prediction = net(G, inputs)
        loss = F.cross_entropy(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_loss = loss.detach().item()
        epoch_loss += iter_loss
        print('epoch {}, iter {}/{}, loss {:.4f}'.format(epoch, iter, len(dataLoader), iter_loss))
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)
    torch.save({'epoch':epoch, 'net_state_dict':net.state_dict(), 'optimizer':optimizer}, './ckpt/skeleton_{}.ckpt'.format(epoch))

