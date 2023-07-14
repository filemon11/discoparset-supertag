import torch.nn as nn
import torch

class Dropout(nn.Module):
    def __init__(self, p):
        super(Dropout, self).__init__()
        self.N = int(10e5)
        self.mask = torch.zeros(self.N, dtype=torch.uint8)
        self.rand = torch.rand(self.N)
        self.p = p

    def forward(self, input_m):
        if self.training and self.p > 0:
            num = input_m.numel()
            if num > self.N:
                self.N = int(num*1.2)
                del self.mask
                del self.rand
                self.mask = torch.zeros(self.N, dtype=torch.uint8)
                self.rand = torch.rand(self.N)

            self.mask[:num] = self.rand[:num].uniform_(0,1) > self.p
#            print(self.mask[:10])
#            print(input_m.view(-1)[:10])
            input_m = input_m * self.mask[:num].view_as(input_m).type(input_m.type())
#            print(input_m.view(-1)[:10])

            #input_m[self.mask[:num].view_as(input_m)] *= 0
            return input_m / (1- self.p)
        return input_m

