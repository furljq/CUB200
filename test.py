from dataLoader import cropCUB as CUB
import pickle
import torch
from graph import build_graph_skeleton as build_graph
from model import GCN as GCN

testset = CUB('test', False)
dataloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)

net = GCN(2048, [1024, 512], 200)
G = build_graph()
ckpt = torch.load('/Disk5/junqi/CUB/early_skeleton_104.ckpt')
net.load_state_dict(ckpt['net_state_dict'])
net = net.cuda()
net = torch.nn.DataParallel(net)
creterion = torch.nn.CrossEntropyLoss()

test_correct = 0
test_loss = 0
total = 0
net.eval()

pre, lab = [], []

for i, (inputs, mask, labels) in enumerate(dataloader):
    if i % 10 == 0 and i > 0:
        print('test {}/{}, acc: {:.3f}, loss: {:.3f}'.format(i, len(dataloader), float(test_correct) / total, test_loss / total))
    with torch.no_grad():
        inputs = inputs.cuda()
        mask = mask.cuda()
        labels = labels.cuda()
        batch_size = inputs.size(0)
        prediction = net(G, inputs, mask)
        pre.append(prediction)
        lab.append(labels)
        loss = creterion(prediction, labels)
        _, predict = torch.max(prediction, 1)
        total += batch_size
        test_loss += loss.item() * batch_size
        test_correct += torch.sum(predict.data == labels.data)

print('total, acc: {:.3f}, loss: {:.3f}'.format(float(test_correct) / total, test_loss / total))

pre = torch.cat(pre, 0)
lab = torch.cat(lab, 0)
pickle.dump(pre, open('./result/early_skeleton.pkl', 'wb'))
pickle.dump(lab, open('./result/labels.pkl', 'wb'))
