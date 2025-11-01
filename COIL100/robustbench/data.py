import os
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# 移除未使用且会导致外部依赖错误的导入（本地不含 model_zoo 等子包）






def load_multiview(n_examples,shuffle,dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    一次性取出多视图数据的一个“大批次”（用于本项目评估/适配）。
    - n_examples: 期望装载的样本数（会作为 DataLoader 的 batch_size）
    - shuffle: 是否打乱
    - dataset: 多视图数据集（其 __getitem__ 返回 [views], target）
    返回：
    - x_test: 张量列表（第 0 维为视角索引），在上层会以 x_test[view] 选择当前视角
    - y_test: 对应标签张量
    """




    # Windows 上多进程 DataLoader 有问题，使用单进程模式
    # pin_memory 在 CPU 模式下也不需要
    num_workers = 0 if sys.platform == 'win32' else 8
    pin_memory = False if not torch.cuda.is_available() else True
    
    test_loader = data.DataLoader(dataset, num_workers=num_workers, batch_size=n_examples,
                                  sampler=None, shuffle=shuffle, pin_memory=pin_memory,
                                    drop_last=True)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test




