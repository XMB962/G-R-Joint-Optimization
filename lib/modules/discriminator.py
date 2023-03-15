import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channel):
        super(Discriminator, self).__init__()
        output_channel = [16, 32, 64]
        self.Conv0 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel[0], 3, 2, 1), 
            nn.BatchNorm2d(output_channel[0]), 
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(output_channel[0], output_channel[1], 3, 2, 1), 
            nn.BatchNorm2d(output_channel[1]), 
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.Conv1 = nn.Sequential(
            nn.Conv2d(output_channel[1], output_channel[2], 3, 2, 1),
            nn.BatchNorm2d(output_channel[2]),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.Linear = nn.Sequential(
            nn.Linear(16*64, 256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(0.5)

    def forward(self, im):
        f = self.Conv0(im)
        f = self.drop(f)
        f = self.Conv1(f)
        f = f.view(f.size(0), -1)
        output = self.Linear(f) + 1e-9
        return -1 * torch.log(output)
