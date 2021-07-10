from torch import nn


class Conv2dSamePadding(nn.Conv1d):
    def __init__(self,*args,**kwargs):
        super(self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(functools.reduce(operator.__add__,
                  [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)