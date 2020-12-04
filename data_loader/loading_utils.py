import torch
import collections
import matplotlib.pyplot as plt


def _get_max_length(x):
    return len(max(x, key=len))


def _pad_sequence(seq):
    def _pad(_it, _max_len):
        return [0] * (_max_len - len(_it)) + _it

    return [_pad(it, _get_max_length(seq)) for it in seq]


def _custom_collate(batch):
    """
    Pads records with a smaller amount of samples with zeros
    :param batch: List[Tuple[Record, Classes, Length]], where Record is a ndarray,  Classes a list and Length an int
    :return:
    """
    records, labels, lengths, record_names = zip(*batch)
    max_length = max(lengths)

    # Convert numpy arrays to tensors and pad shorter records with 0
    records = [torch.from_numpy(record) for record in records]
    padded_records = []
    for record in records:
        target = torch.zeros(12, max_length)
        target[:, :len(record[0])] = record
        padded_records.append(target)

        # Plot visualization for the first lead
        # fig, axs = plt.subplots(2, 1, figsize=(15,15))
        # axs[0].plot(record[0].numpy())
        # axs[0].set_title("Unpadded first lead")
        # axs[1].plot(target[0].numpy())
        # axs[1].set_title("Padded first lead")
        # plt.show()

    return torch.stack(padded_records), labels, lengths, record_names