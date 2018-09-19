import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'temporal':
        from data.temporal_dataset import TemporalDataset
        dataset = TemporalDataset()   
    elif opt.dataset_mode == 'face':
        from data.face_dataset import FaceDataset
        dataset = FaceDataset() 
    elif opt.dataset_mode == 'pose':
        from data.pose_dataset import PoseDataset
        dataset = PoseDataset() 
    elif opt.dataset_mode == 'test':
        from data.test_dataset import TestDataset
        dataset = TestDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
