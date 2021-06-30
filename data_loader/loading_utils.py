import math

import pandas as pd
#  collate_fn receives a list of tuples if your __getitem__ function from a Dataset subclass returns a tuple
import torch


def _collate_pad_or_truncate(batch, seq_len):
    """
    Pads records with a smaller amount of samples with zeros
    Cuts records that exceed seq_len from both sides and only uses values in the middle
    :param batch: List[Tuple[Record, Classes, Length, Record_name]],
            where Record is a dataframe,  classes a list of ints, length an int and record name a string
    :return:
    """
    records, labels, labels_one_hot, lengths, record_names = zip(*batch)
    # Pad records shorter than seq_len with 0, clip longer ones
    seq_len_records = []
    for idx, df_record in enumerate(records):
        record_len = len(df_record.index)
        diff = seq_len - record_len

        # Plot record to verify the padding visually
        #plot_record_from_df(record_name=record_names[idx], df_record=df_record, preprocesed=False)

        if diff > 0:
            # Pad the record to the maximum length of the batch
            df_zeros = pd.DataFrame([[0] * df_record.shape[1]] * diff, columns=df_record.columns)
            df_record = pd.concat([df_zeros, df_record], axis=0, ignore_index=True)
        elif diff < 0:
            # Cut the record to have length seq_len (if possible, cut the equal amount of values from both sides)
            # If the diff is not even, cut one value more from the beginning
            df_record = df_record.iloc[math.ceil(-diff/2):record_len-math.floor(-diff/2)]

        # Plot record to verify the padding visually
        #plot_record_from_df(record_name=record_names[idx], df_record=df_record, preprocesed=True)

        # Convert the df to a numpy array before appending it to the list
        # Like this, the conversion to tensors is automatically handled by the dataloader
        seq_len_records.append(df_record.values)

    # TODO DEAL WITH MULTI_LABEL CASE, at the moment only the first label is used per record
    return torch.tensor(seq_len_records).float(), torch.tensor([label[0] for label in labels]), \
           torch.tensor(labels_one_hot), lengths, record_names
