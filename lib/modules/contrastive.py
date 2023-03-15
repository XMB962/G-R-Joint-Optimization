from turtle import forward
import torch
import torch.nn as nn

class contrastive_mapping(nn.Module):
    def __init__(self, input_num, hidden_num, ):
        super(contrastive_mapping, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )
    def forward(self, x):
        return self.ffn(x)




