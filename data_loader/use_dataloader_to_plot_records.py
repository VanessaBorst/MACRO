import argparse
import collections

import matplotlib.pyplot as plt

import data_loader.ecg_data_loader as ecg_loader
from parse_config import ConfigParser


def _test_dataloader(config_parser):
    # dirname = os.path.dirname(__file__)
    # input_directory = os.path.join(dirname, '../data/CPSC/raw/Training_WFDB')
    # ecg_data_loader = ECGDataLoader(input_directory, 4, True)

    # setup data_loader instance
    ecg_data_loader = config_parser.init_obj('data_loader', ecg_loader)

    for batch_num, batch_data in enumerate(ecg_data_loader):
        # padded_records, labels, lengths, record_names = zip(*batch_data)
        padded_records = batch_data[0]
        labels = batch_data[1]
        # lengths = batch_data[2]
        record_names = batch_data[3]
        print(padded_records.shape)
        print(labels)

        for i in range(len(padded_records)):
            plt.plot(padded_records[i][0])
            plt.title("Lead I of record " + str(record_names[i])+ " of batch " + str(batch_num) + ": Label = " + str(labels[i]))
            plt.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='MA')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config_parser = ConfigParser.from_args(args, options)
    _test_dataloader(config_parser)