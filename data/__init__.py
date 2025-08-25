import importlib
from data.base_dataset import BaseDataset
from torch.utils.data import DataLoader


def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = f"{dataset_name}Dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))
    return dataset


def build_dataloader(cfg,phase):
    Dataset = find_dataset_using_name(cfg.dataset.name)
    data = Dataset()
    data.initialize(cfg,phase)

    if phase == 'train':
         dataloader = DataLoader(data, batch_size=cfg.model.batch_size, shuffle=True,
                   num_workers=cfg.model.batch_size * 2, pin_memory=True)
    elif phase == 'test':
        dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    else:
        dataloader = DataLoader(data, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader,data
