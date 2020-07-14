import os

import torch
import torch.nn as nn
import torch.utils.data as data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleTensorDataset(data.Dataset):
    def __init__(self, file):
        self.tensor = torch.load(file)

    def __getitem__(self, item):
        return self.tensor[item]

    def __len__(self):
        return self.tensor.size(0)


def data_iter(file, batch_size):
    return data.DataLoader(
        dataset=SingleTensorDataset(file=file),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        # num_workers=2,
    )


def prepare_gru_inputs(input, sos=0):
    """
    :param input: batch_size x seq_len
    :param sos:
    :return:
    """
    inp = torch.ones_like(input) * sos
    inp[:, 1:] = input[:, :input.size(1) - 1]
    return inp


def nll_loss(oracle_net, target):
    inp = prepare_gru_inputs(target, oracle_net.sos)
    out, _ = oracle_net(inp)
    # nll = nn.NLLLoss(reduction='elementwise_mean')
    loss = nn.NLLLoss()(out.view(-1, out.size(-1)), target.view(-1))
    return loss.mean()


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, rate=0.8):
    dic = {}
    for name, param in source.named_parameters():
        dic[name] = param.data
    for name, param in target.named_parameters():
        if name.startswith('emb'):
            param.data.copy_(dic[name])
        else:
            param.data.copy_(rate * param.data + (1 - rate) * dic[name])


def get_accuracy(net, real, fake):
    real_score = net(real)
    fake_score = net(fake)
    acc = 0.5 * ((real_score > 0.5).float().mean() + (fake_score <= 0.5).float().mean())
    return acc.item()
