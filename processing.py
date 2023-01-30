import json
from typing import List, Union

import torch

# torch.manual_seed(1332)


def get_config():
    with open("./config/paths.json", "r") as f:
        config = json.load(f)
    return config


def read_txt_dataset(path: str = None):
    if path is None:
        c = get_config()
        path = c.get('dataset')
    with open(path, "r") as f:
        data = f.read()
    return data


class Vocab:

    def __init__(self, corpus: str):
        self.vocab = list(set(corpus))
        self.size = len(self.vocab)
        self.D = {v: i for i, v, in enumerate(self.vocab)}
        self.D_T = {i: v for v, i in self.D.items()}

    def encode(self, doc: str):
        return [self.D[char] for char in doc]

    def decode(self, ids: Union[List[List[int]], List[int]]):
        return ''.join([self.D_T[i] for i in ids])


def get_batch(data, block_size):
    ix = torch.randint(len(data) - block_size, (block_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i+block_size + 1] for i in ix])
    return x, y

