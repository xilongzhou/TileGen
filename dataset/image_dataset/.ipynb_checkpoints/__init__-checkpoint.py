from .scene_dataset import Dataset as SceneDataset 
from .refiner_dataset import Dataset as RefinerDataset 
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



def get_refiner_dataloader(args, train):   
    dataset = RefinerDataset(args,train)   
    sampler = data_sampler(args, dataset)
    return data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)



# Composition do not have dataloader