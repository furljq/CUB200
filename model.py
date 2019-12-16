import torch
import dgl
import torch.nn as nn
from resnet import resnet50


def gcn_message(edges):
    return {'msg': edges.src['h']}


def gcn_reduce(nodes):
    return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}


class GCNLayer(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        g.ndata['h'] = inputs
        g.send(g.edges(), gcn_message)
        g.recv(g.nodes(), gcn_reduce)
        h = g.ndata.pop('h')
        return self.linear(h.float())


class GCN(nn.Module):

    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = nn.Linear(512 * 4, 200)
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.8)
        self.gcn2 = GCNLayer(hidden_size, hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.7)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, G, mask, feat):
        feat = feat.view(feat.shape[0] * 15, feat.shape[2], feat.shape[3], feat.shape[4])
        h = self.resnet(feat.permute(0,3,1,2))
        h = h.view(-1, 15, h.shape[1]).permute(1,0,2)
        h[~mask] = torch.zeros(h.shape[2])
        h = self.gcn1(G, h)
        h = h.permute(1,2,0)
        h = self.norm1(h)
        h = h.permute(2,0,1)
        h = torch.relu(h)
        h = self.dropout1(h)
        h = self.gcn2(G, h)
        h = h.permute(1,2,0)
        h = self.norm2(h)
        h = h.permute(2,0,1)
        h = torch.relu(h)
        h = self.dropout2(h)
        G.ndata['h'] = h
        h = dgl.mean_nodes(G, 'h')
        return self.classifier(h)


