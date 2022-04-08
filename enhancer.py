import torch.nn as nn
import torch
import torch.nn.functional as F

class Enhancer(nn.Module):
    def __init__(self,
                 input_size,
                 num_cells,
                 hidden_size,
                 dropout=0.) -> None:
        super(Enhancer, self).__init__()
        self.num_cells = num_cells

        self.rnns = nn.ModuleList()
        for i in range(self.num_cells):
            rnn = nn.GRUCell(input_size, hidden_size)
            self.rnns.append(rnn)
        
        self.hidden2output = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)

        need_init = [self.rnns]
        for layer in need_init:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.orthogonal_(p)

        need_init = [self.hidden2output]
        for layer in need_init:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, current, history, init_hidden):
        currents = torch.split(current, 1, dim=1)
        history = history.view(history.size(0), -1, history.size(-1))
        outputs = []
        hiddens = []
        hidden = init_hidden
        input = currents[0]
        for index, gruCell in enumerate(self.rnns):
            input = currents[index]
            input = input.squeeze(1)
            hidden = gruCell(input, hidden)
            hiddens.append(hidden)
            output_embed = F.tanh(self.dropout(self.hidden2output(hidden)))
            
            outputs.append(output_embed)
        
        return torch.stack(outputs, dim=1), torch.stack(hiddens, dim=1)