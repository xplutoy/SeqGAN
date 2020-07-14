import torch.nn.functional as F
import torch.nn.init as init

from beam import batch_beam
from utils import *


class Generator(nn.Module):
    def __init__(self, voc_size, emb_size, hid_size, max_seq_len=20, sos=0, oracle_init=False):
        super(Generator, self).__init__()
        self.max_seq_len = max_seq_len
        self.sos = sos

        self.embedding = nn.Embedding(voc_size, emb_size)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)
        self.fc_out = nn.Linear(hid_size, voc_size)

        if oracle_init:
            for p in self.parameters():
                init.normal_(p, 0, 1)

        self.to(DEVICE)

    def forward(self, input, hidden=None):
        emb = self.embedding(input)
        out, hidden = self.gru(emb, hidden)
        out = self.fc_out(out)
        out = F.log_softmax(out, -1)

        return out, hidden

    def sample_and_logprobs(self, num, k=10, method=None):
        if method == 'beam':
            return self.beam_sample(num, k)
        else:
            return self.rand_sample(num)

    def beam_sample(self, num, k):
        return batch_beam(self, num, k)

    def rand_sample(self, num):
        output = []
        selected_log_probs = []
        hidden = None
        input = self.sos * torch.ones(num, 1).long().to(DEVICE)
        for t in range(self.max_seq_len):
            rnn_out, hidden = self.forward(input, hidden)
            rnn_out = rnn_out.squeeze()
            input = torch.multinomial(torch.exp(rnn_out), 1)
            log_prob = rnn_out.gather(-1, input)
            output.append(input)
            selected_log_probs.append(log_prob)
        return torch.cat(output, -1), torch.cat(selected_log_probs, -1)

    def roll_out(self, input):
        """
        :param input: batch_size x seq_len
        :return:
        """
        seq_len = input.size(1)
        rollouts = []

        hidden = None
        for t in range(seq_len - 1):
            out = [input[:, :t + 1]]
            inp = prepare_gru_inputs(input)

            for i in range(t + 2):
                rnn_out, hidden = self.forward(inp[:, i].unsqueeze(1), hidden)
            pred = torch.multinomial(torch.exp(rnn_out.squeeze()), 1)
            out.append(pred)
            for j in range(seq_len - t - 2):
                rnn_out, hidden = self.forward(pred, hidden)
                pred = torch.multinomial(torch.exp(rnn_out.squeeze()), 1)
                out.append(pred)

            out = torch.cat(out, 1)
            rollouts.append(out)
        rollouts.append(input)
        return rollouts
