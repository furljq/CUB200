from dataLoader import cropCUB
import torch
from graph import build_graph
from model import GCN

testset = cropCUB('test', False)
dataloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)

net = GCN(200, 500, 200)
G = build_graph()
ckpt = torch.load('./ckpt/model_19.ckpt')
net.load_state_dict(ckpt['net_state_dict'])
net = net.cuda()
net = torch.nn.DataParallel(net)
creterion = torch.nn.CrossEntropyLoss()

test_correct = 0
test_loss = 0
total = 0
net.eval()

for i, (inputs, labels) in enumerate(dataloader):
    if i % 10 == 0 and i > 0:
        print('test {}/{}, acc: {:.3f}, loss: {:.3f}'.format(i, len(dataloader), float(test_correct) / total, test_loss / total))
    with torch.no_grad():
        inputs = inputs.cuda()
        labels = labels.cuda()
        batch_size = inputs.size(0)
        prediction = net(G, inputs)
        loss = creterion(prediction, labels)
        _, predict = torch.max(prediction, 1)
        total += batch_size
        test_loss += loss.item() * batch_size
        test_correct += torch.sum(predict.data == labels.data)

print('total, acc: {:.3f}, loss: {:.3f}'.format(i, len(dataloader), float(test_correct) / total, test_loss / total))
