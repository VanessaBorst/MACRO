import torch
from torch.nn import BCELoss, BCEWithLogitsLoss, MultiLabelSoftMarginLoss
from torch.autograd import Variable

x = Variable(torch.randn(10, 3))
y = Variable(torch.FloatTensor(10, 3).random_(2))

# double the loss for class 1
class_weight = torch.FloatTensor([1.0, 2.0, 1.0])
# double the loss for last sample
element_weight = torch.FloatTensor([1.0]*9 + [2.0]).view(-1, 1)
element_weight = element_weight.repeat(1, 3)

bce_criterion = BCEWithLogitsLoss(weight=None, reduce=False)
multi_criterion = MultiLabelSoftMarginLoss(weight=None, reduce=False)

bce_criterion_class = BCEWithLogitsLoss(weight=class_weight, reduce=False)
multi_criterion_class = MultiLabelSoftMarginLoss(weight=class_weight, reduce=False)

bce_criterion_element = BCEWithLogitsLoss(weight=element_weight, reduce=False)
multi_criterion_element = MultiLabelSoftMarginLoss(weight=element_weight, reduce=False)

bce_loss = bce_criterion(x, y)
multi_loss = multi_criterion(x, y)

bce_loss_class = bce_criterion_class(x, y)
multi_loss_class = multi_criterion_class(x, y)

bce_loss_element = bce_criterion_element(x, y)
multi_loss_element = multi_criterion_element(x, y)

print(bce_loss - multi_loss)
print(bce_loss_class - multi_loss_class)
print(bce_loss_element - multi_loss_element)