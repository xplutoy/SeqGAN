import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DEVICE


# borrow from https://github.com/ZiJianZhao/SeqGAN-PyTorch/blob/master/discriminator.py
class CNNDiscriminator(nn.Module):
    def __init__(self, voc_size, emb_size=64, dropout_prob=0.5,
                 filter_sizes=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20),
                 num_filters=(100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160)):
        super(CNNDiscriminator, self).__init__()
        self.embedding = nn.Embedding(voc_size, emb_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filter, kernel_size=(filter_size, emb_size)) for filter_size, num_filter in
             zip(filter_sizes, num_filters)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(sum(num_filters), 1)
        self.init_parameters()
        self.to(DEVICE)

    def init_parameters(self):  # 没有这个初始化就会出现饱和的现象
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, x):
        embed = self.embedding(x).unsqueeze(1)  # (N, 1, sequence_length, embedding_size)
        convs = [F.relu(conv(embed)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)
        highway = self.highway(pred)
        pred = F.sigmoid(highway) * F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = F.sigmoid(self.linear(self.dropout(pred)))
        return pred

    def rewards(self, rolls):
        rets = []
        for r in rolls:
            ret_t = self.forward(r)
            rets.append(ret_t)
        return torch.cat(rets, 1)
