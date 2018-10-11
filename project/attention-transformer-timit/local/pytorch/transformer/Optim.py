'''A wrapper class for optimizer '''
import numpy as np

class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, start_lr = 0.001, soft_coefficient = 500):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.soft_coefficient = soft_coefficient
        self.n_current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        new_lr = (self.start_lr * self.soft_coefficient) / (self.n_current_steps + self.soft_coefficient)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
