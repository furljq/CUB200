import pickle
import torch

pre1 = pickle.load(open('./result/resnet.pkl', 'rb'))
pre2 = pickle.load(open('./result/early_skeleton_753.pkl', 'rb'))
labels = pickle.load(open('./result/labels.pkl', 'rb'))

prediction = pre1 + pre2
_, predict = torch.max(prediction, 1)
correct = torch.sum(predict.data == labels.data)
correct = float(correct) / prediction.shape[0]

print(correct)
