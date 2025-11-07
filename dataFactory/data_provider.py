
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .LazyPickleDataset import LazyPickleDataset
from .collate_fn import collate_fn

def data_provider(args, flag, mode=None):
    ################################################ Dataset 初始化
    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True

    drop_last = False
    batch_size = args.micro_batch# 使用 micro_batch（单GPU的batch size）

    if mode == 'type':
        data_set = LazyPickleDataset(
            pkl_path=f'{args.root_path}/type_{flag}.pkl',
            index_path=f'{args.root_path}/type_{flag}.index'
        )
    else:
        data_set = LazyPickleDataset(
            pkl_path=f'{args.root_path}/process_data_{flag}.pkl',
            index_path=f'{args.root_path}/process_data_{flag}.index'
        )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

    return data_set, data_loader
