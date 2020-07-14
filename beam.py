import torch

from utils import DEVICE


def beam(net, k):
    outputs = None
    selected_log_probs = None
    hidden = None
    input = net.sos * torch.ones(1, 1).long().to(DEVICE)

    for t in range(net.max_seq_len):
        rnn_out, hidden = net(input, hidden)
        rnn_out = rnn_out.squeeze(1)
        val, idx = rnn_out.topk(k, dim=-1, sorted=False)
        if t == 0:
            selected_log_probs = val.t()
            outputs = idx.t()
            hidden = hidden.repeat(1, k, 1)
        else:
            pre_score = selected_log_probs.sum(1, keepdim=True).repeat(1, k)  # k x k
            cur_score = (pre_score + val).view(1, -1)
            score, score_idx = cur_score.topk(k, sorted=False)
            selected_val = val.view(1, -1).gather(1, score_idx)
            selected_idx = idx.view(1, -1).gather(1, score_idx)
            pre_idx = (score_idx / k).long().squeeze()
            selected_log_probs = torch.cat([selected_log_probs.index_select(0, pre_idx), selected_val.t()], -1)
            outputs = torch.cat([outputs.index_select(0, pre_idx), selected_idx.t()], -1)

        input = outputs[:, -1].unsqueeze(1)

    return outputs, selected_log_probs


def batch_beam(net, batch_size, k):
    outputs = []
    selected_log_probs = []
    for bs in range(batch_size):
        k_outs, k_log_probs = beam(net, k)
        # 选取总体分最大的
        total_score, total_idx = k_log_probs.sum(1).topk(1)
        outputs.append(k_outs[total_idx, :])
        selected_log_probs.append(k_log_probs[total_idx, :])
    return torch.cat(outputs), torch.cat(selected_log_probs)
