import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss


# ------------------------- Multi-Label Classification Loss Function -------------------------
def balanced_BCE(output, target, class_weights=None):
    # TODO set class weights correctly, e.g. weight=class_weight[target.long()]
    # loss = BCELoss(weight=class_weight[target.long()])
    loss = BCELoss(weight=class_weights)
    return loss(output, target)


def BCE(output, target, class_weights=None):
    # loss = BCELoss(weight=class_weight[target.long()])
    loss = BCELoss()
    return loss(output, target.float())


# This contains Sigmoid itself
def BCE_with_logits(output, target):
    loss = BCEWithLogitsLoss()
    return loss(output, target)


# This contains Sigmoid itself
def balanced_BCE_with_logits(output, target, class_weight=None):
    # TODO set positive weights correctly, e.g. pos_weight=class_weight[1]
    loss = BCEWithLogitsLoss(pos_weight=None)
    return loss(output, target)


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


def cross_entropy_loss(output,target):
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