
import torch
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss, MultiLabelSoftMarginLoss

from loss.asymmetric_loss import AsymmetricLoss, AsymmetricLossOptimized


# ------------------------- Multi-Label Classification Loss Function -------------------------
def balanced_BCE(output, target, class_weights):
    # TODO set class weights correctly, e.g. weight=class_weight[target.long()]
    loss = BCELoss(weight=torch.Tensor(class_weights))
    return loss(output, target.float())


def multi_label_soft_margin(output, target, class_weights):
    # TODO set class weights correctly, e.g. weight=class_weight[target.long()]
    loss = MultiLabelSoftMarginLoss(weight=torch.Tensor(class_weights))
    return loss(output, target)


# def BCE(output, target):
#     loss = BCELoss()
#     return loss(output, target.float())

# Code from https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained
def focal_binary_cross_entropy_with_logits(output, target, gamma=2, beta=0.25):
    num_label = target.size(1)
    l = output.reshape(-1)
    t = target.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where(t >= 0.5, p, 1 - p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
    loss = beta * logp * ((1 - p) ** gamma)
    loss = num_label * loss.mean()
    return loss


def asymmetric_loss(output, target, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
    loss = AsymmetricLossOptimized(gamma_neg, gamma_pos, clip, eps, disable_torch_grad_focal_loss)
    return loss(output, target)


# This contains Sigmoid itself
def BCE_with_logits(output, target):
    loss = BCEWithLogitsLoss()
    return loss(output, target.float())


def multi_branch_asymmetric_loss_with_logits(output, target, single_lead_outputs, lambda_balance,
                                             gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                                             disable_torch_grad_focal_loss=True):
    loss_fn = AsymmetricLossOptimized(gamma_neg, gamma_pos, clip, eps, disable_torch_grad_focal_loss)
    # Calculate loss for each branch
    sum_single_branch_losses = 0
    for single_lead_output in single_lead_outputs:
        sum_single_branch_losses += loss_fn(single_lead_output, target.float())
    # Calculate loss for concatenated multi-branched output
    multi_branch_loss = loss_fn(output, target.float())
    # Calculate the joint loss of each single lead branch and the overall network (weighted sum)
    return multi_branch_loss + lambda_balance * sum_single_branch_losses

def multi_branch_BCE_with_logits(output, target, single_lead_outputs, lambda_balance):
    loss_fn = BCEWithLogitsLoss()
    # Calculate loss for each branch
    sum_single_branch_losses = 0
    for single_lead_output in single_lead_outputs:
        sum_single_branch_losses += loss_fn(single_lead_output, target.float())
    # Calculate loss for concatenated multi-branched output
    multi_branch_loss = loss_fn(output, target.float())
    # Calculate the joint loss of each single lead branch and the overall network (weighted sum)
    return multi_branch_loss + lambda_balance * sum_single_branch_losses


# This contains Sigmoid itself
def balanced_BCE_with_logits(output, target, pos_weights, device):
    pos_weights_tensor = torch.tensor(pos_weights).to(device)
    loss = BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
    return loss(output, target.float())


# ------------------------- Single-Label Classification Loss Functions -------------------------
def nll_loss(output, target):
    """
     The negative log likelihood loss to train a classification problem with C classes

     Parameters
     ----------
     :param output: dimension=(minibatch,C);
        Per entry, the log-probabilities of each class should be contained,
        obtaining log-probabilities is achieved by adding a LogSoftmax layer in the last layer of the network
     :param target: dimension= (N)
        Per entry, a class index in the range [0, C-1] as integer

    Comments
    ----------
    Alternative: You may use CrossEntropyLoss instead, if you prefer not to add an extra layer
        torch.nn.CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class
    """
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    """
         The cross entropy loss to train a classification problem with C classes

         Parameters
         ----------
         :param output: dimension=(minibatch,C);
            Per entry, the raw, unnormalized scores for each class should be contained
         :param target: dimension= (N)
            Per entry, a class index in the range [0, C-1] as integer
        """
    return F.cross_entropy(output, target)


def balanced_cross_entropy(output, target, class_weights, device):
    """
     The cross entropy loss to train a classification problem with C classes

     Parameters
     ----------
     :param output: dimension=(minibatch,C);
        Per entry, the raw, unnormalized scores for each class should be contained
     :param target: dimension= (N)
        Per entry, a class index in the range [0, C-1] as integer
    :param class_weights:  a manual rescaling weight given to each class
    :param device: device currently used
    """
    class_weights_tensor = torch.tensor(class_weights).to(device).float()
    return F.cross_entropy(output, target, class_weights_tensor)

# Balanced cross entropy: Usually you increase the weight for minority classes, so that their loss also increases and
# forces the model to learn these samples. This could be done by e.g. the inverse class count (class frequency).:
# https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/22

# Poss. 1:  Weight of class c is the size of largest class divided by the size of class c.
# You can also use the smallest class as nominator; this is only a re-scaling, the relative weights are the same
# (https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch)
# Poss. 2: It could be done by e.g. the inverse class count (class frequency)
# https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/23
