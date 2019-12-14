import numpy as np
import torch
import torch.nn.functional as F
from model import GCN
from graph import build_graph
from dataLoader import cropCUB

net = GCN(200, 500, 200).cuda()
G = build_graph()

datas = cropCUB()
dataLoader = torch.utils.data.DataLoader(datas, batch_size=8, shuffle=True, num_workers=4)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

epoch_losses = []
for epoch in range(80):
    epoch_loss = 0
    torch.save({'epoch':epoch, 'net_state_dict':net.state_dict(), 'optimizer':optimizer}, '../ckpt/model_{}.ckpt'.format(epoch))
    for iter, (inputs, labels) in enumerate(dataLoader):
        inputs = inputs.cuda()
        labels = labels.cuda()
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

