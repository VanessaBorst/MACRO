import torch.nn.functional as F


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

