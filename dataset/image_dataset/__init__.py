from .scene_dataset import Dataset as SceneDataset 
import os 
import time 
import torch 
from torch.utils import data


def data_sampler(args, dataset):
    if args.distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=args.shuffle, seed=args.sampler_seed)
    if args.shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)




def get_scene_dataloader(args, train):   
    dataset = SceneDataset(args,train)   
    sampler = data_sampler(args, dataset)
    return data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)



