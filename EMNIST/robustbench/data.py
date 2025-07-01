from typing import  Tuple
import torch
import torch.utils.data as data


def load_multiview(n_examples,shuffle,dataset) -> Tuple[torch.Tensor, torch.Tensor]:




    test_loader = data.DataLoader(dataset, num_workers=8, batch_size=n_examples,
                                  sampler= None,shuffle=shuffle,pin_memory=True,
                                    drop_last=True)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test


