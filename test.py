from dataLoader import cropCUB
import torch
from graph import build_graph
from model import GCN

testset = cropCUB('test')
dataloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)

net = GCN(200, 500, 200)
G = build_graph()
ckpt = torch.load('../ckpt/model_1.ckpt')
net.load_state_dict(ckpt['net_state_dict'])
net = net.cuda()
net = torch.nn.DataParallel(net)

test_correct = 0
total = 0
for i, (inputs, labels) in enumerate(dataloader):
    if i % 10 == 0 and i > 0:
        print('test {}/{}, score: {}'.format(i, len(dataloader), float(test_correct) / total))
    with torch.no_grad():
        inputs = inputs.cuda()
        labels = labels.cuda()
        batch_size = inputs.size(0)
        prediction = net(G, inputs)
        _, predict = torch.max(prediction, 1)
        total += batch_size
        test_correct += torch.sum(predict.data == labels.data)

