import torch
from tqdm import tqdm
from get_data import train_loader

sum_mean, sum_std = torch.tensor([0., 0., 0.]), torch.tensor([0., 0., 0.])

for i, (train_imgs, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
    for k in range(3):
        sum_mean[k] += train_imgs[:, k, :, :].mean()
        sum_std[k] += train_imgs[:, k, :, :].std()

mean = sum_mean / len(train_loader)
std = sum_std / len(train_loader)

print(mean, std)