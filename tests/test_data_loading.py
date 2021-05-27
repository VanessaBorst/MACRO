import data_loader.ecg_data_loader as ecg_loader
import matplotlib.pyplot as plt


# def test_fixture_plugins(a_plugin_fixture, yet_some_other_plugin_fixture):
#     assert a_plugin_fixture == 'a_plugin_fixture'
#     assert yet_some_other_plugin_fixture == 'yet_some_other_plugin_fixture'


def test_dataloader(default_config_parser):
    # setup data_loader instance
    ecg_data_loader = default_config_parser.init_obj('data_loader', ecg_loader)
    # assert ecg_data_loader.batch_size == default_config_parser.config["data_loader"]["args"]["batch_size"]

    for batch_num, batch_data in enumerate(ecg_data_loader):
        padded_records = batch_data[0]
        labels = batch_data[1]
        lengths = batch_data[2]
        record_names = batch_data[3]

        if not batch_num == len(ecg_data_loader) - 1:
            # The last batch could be smaller
            assert len(padded_records) == len(labels) == len(lengths) == len(record_names) == \
                   default_config_parser.config["data_loader"]["args"]["batch_size"]
        else:
            # If drop Last is True, then the final batch needs to be of size "batch size" as well
            assert ecg_data_loader.drop_last == False or len(padded_records) == len(labels) == len(lengths) == \
                   len(record_names) == default_config_parser.config["data_loader"]["args"]["batch_size"]
