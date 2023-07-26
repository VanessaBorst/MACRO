import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.01)


def plot(func, yaxis=(-1.4, 1.4), title=""):
    f = plt.figure()
    plt.ylim(yaxis)
    plt.locator_params(nbins=3)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.axhline(lw=1, c='black')
    plt.axvline(lw=1, c='black')
    plt.grid(alpha=0.4, ls='-.')
    plt.box(on=None)
    plt.plot(x, func(x), c='r', lw=3)
    f.savefig("plots/" + title + ".pdf", bbox_inches='tight')
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    #return 2 / (1 + np.exp(-2 * x)) -1
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


binary_step = np.vectorize(lambda x: 1 if x > 0 else 0, otypes=[np.float])
plot(binary_step, yaxis=(-0.2, 1.4), title="BinaryStep")

plot(sigmoid, yaxis=(-0.2, 1.4), title="Sigmoid")

relu = np.vectorize(lambda x: x if x > 0 else 0, otypes=[np.float])
plot(relu, yaxis=(-2, 5), title="ReLU")

leaky_relu = np.vectorize(lambda x: max(0.1 * x, x), otypes=[np.float])
plot(leaky_relu, yaxis=(-2, 5), title="LeakyReLU")

plot(tanh, yaxis=(-1.4, 1.4), title="tanH")

hard_tanh = np.vectorize(lambda x: max(min(x,1), -1), otypes=[np.float])
plot(hard_tanh, yaxis=(-1.4, 1.4), title="HardTanH")