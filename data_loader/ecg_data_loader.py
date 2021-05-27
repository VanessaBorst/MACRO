from base import BaseDataLoader


from data_loader.ecg_data_set import ECGDataset
from data_loader.loading_utils import _custom_collate
from torchvision import transforms


class ECGDataLoader(BaseDataLoader):
    """
    ECG data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        trsfm = None
        transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0,), (1,))      # TODO Check if this works for non-image data -> probably not
          ])
        self.data_dir = data_dir
        self.dataset = ECGDataset(self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, _custom_collate)


