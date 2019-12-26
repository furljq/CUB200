import numpy as np
import torch
import torch.nn.functional as F
from model import GCN as GCN
from graph import build_graph_skeleton as build_graph
from dataLoader import cropCUB as CUB

CD = True

G = build_graph()

net = GCN(2048, [1024, 512], 200)
if CD:
    net = net.cuda(0)

batch_size = 8
start_epoch = 0
if True:
    ckpt = torch.load('/Disk5/junqi/CUB/early_skeleton_93.ckpt')
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1

datas = CUB()
dataLoader = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=True, num_workers=4)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

net.train()
for epoch in range(start_epoch, 201):
    epoch_loss = 0
    correct = 0
    for iter, (inputs, mask, labels) in enumerate(dataLoader):
        if CD:
            inputs = inputs.cuda(0)
            mask = mask.cuda(0)
            labels = labels.cuda(0)
        prediction = net(G, inputs, mask)
        _, predict = torch.max(prediction, 1)
        loss = F.cross_entropy(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_loss = loss.detach().item()
        correct += torch.sum(predict.data == labels.data)
        epoch_loss += iter_loss
        print('epoch {}, iter {}/{}, loss {:.4f}'.format(epoch, iter, len(dataLoader), iter_loss))
    epoch_loss /= (iter + 1)
    correct = float(correct) / ((iter + 1) * batch_size)
    print('Epoch {}, loss {:.4f}, acc {:.4f}'.format(epoch, epoch_loss, correct))
    if epoch % 1 == 0:
        torch.save({'epoch':epoch, 'net_state_dict':net.state_dict(), 'optimizer':optimizer}, '/Disk5/junqi/CUB/early_skeleton_{}_{:.4f}_{:.4f}.ckpt'.format(epoch, correct, epoch_loss))

