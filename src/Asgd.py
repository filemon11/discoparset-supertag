import torch
from torch.optim.optimizer import Optimizer


class MyAsgd(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0, 
                 k=-1, gaussian_noise=True, noise=0.01, dc=1e-7):
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)
        super(MyAsgd, self).__init__(params, defaults)
        self.n_steps = 1
        self.k = k
        self.gaussian_noise = gaussian_noise
        self.noise = noise
        self.lr = lr
        self.dc = dc
        #self.warmup = warmup

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['noise'] = torch.zeros_like(p.data)
                param_state['cache'] = torch.zeros_like(p.data)
                param_state['cache'].copy_(p.data)


    def _get_rate(self):
        if self.n_steps > 1000:
            return self.lr / (1 + self.n_steps * self.dc)
        standard = self.lr / (1 + self.n_steps * self.dc)
        return standard * self.n_steps / 1000

    def __setstate__(self, state):
        super(MyAsgd, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            group['lr'] = self._get_rate()
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if self.gaussian_noise:
#                    if 'noise' not in param_state:
#                        param_state['noise'] = torch.zeros_like(p.data)
                    noise = param_state['noise']
                    noise.normal_(0, std=self.noise / (1 + self.n_steps)**0.55)
                    d_p.add_(noise)

                if self.n_steps >= self.k:
#                    if 'cache' not in param_state:
#                        param_state['cache'] = torch.zeros_like(p.data)
#                        param_state['cache'].copy_(p.data)
                    param_state['cache'].add_(-group['lr']*self.n_steps, d_p)
                p.data.add_(-group['lr'], d_p)

        self.n_steps += 1
        return loss


    def average(self):
        if self.n_steps < self.k:
            return
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'saved' not in param_state:
                    param_state['saved'] = torch.zeros_like(p.data)
                param_state['saved'].copy_(p.data)
                if 'cache' in param_state:
                    p.data.add_(-1/self.n_steps, param_state['cache'])

    def cancel_average(self):
        if self.n_steps < self.k:
            return
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['saved'])





