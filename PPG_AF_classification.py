from tsai.all import *
import os
import torchvision.models as models
import torch.nn as nn
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision
from glob import glob
from torchinfo import summary
import numpy as np
import torch.functional as F
import torchvision.transforms as T
from tqdm.auto import tqdm
import torchmetrics
device0 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
batch_size = 64

# fit


def resized(data, N):
    M = data.size
    res = np.empty(N, data.dtype)
    carry = 0
    m = 0
    for n in range(N):
        sum = carry
        while m*N - n*M < M:
            sum += data[m]
            m += 1
        carry = (m-(n+1)*M/N)*data[m-1]
        sum -= carry
        res[n] = sum*N/M
    return res


class CustomDataset(Dataset):
    def __init__(self, data, label, transform=None, target_transform=None):
        self.PPG_data = data
        self.PPG_label = label

    def __len__(self):
        return len(self.PPG_data)

    def __getitem__(self, idx):
        signal_tensor = self.PPG_data[idx]
        AF_signal_label = self.PPG_label[idx]
        return signal_tensor, AF_signal_label


train_csv_path = '../../data/MixArtifacts/b30sec/train/'
train_df = pd.read_csv(
    '../../data/MixArtifacts/b30sec/train.csv', encoding='cp949')

train_data = torch.empty(len(train_df), 1, 3000)
train_label = torch.empty(len(train_data), 1)

for i in tqdm(range(len(train_df))):
    file_name = train_df.loc[i]['FileName']
    AF_signal_label = train_df.loc[i]['AF']
    signal_csv = pd.read_csv(train_csv_path+file_name)['PPG'].to_numpy()
    train_data[i] = torch.tensor(signal_csv)
    train_label[i] = torch.tensor([AF_signal_label])

test_csv_path = '../../data/MixArtifacts/b30sec/test/'
test_df = pd.read_csv(
    '../../data/MixArtifacts/b30sec/test.csv', encoding='cp949')

test_data = torch.empty(len(test_df), 1, 3000)
test_label = torch.empty(len(test_data), 1)

for i in tqdm(range(len(test_df))):
    file_name = test_df.loc[i]['FileName']
    AF_signal_label = test_df.loc[i]['AF']
    signal_csv = pd.read_csv(test_csv_path+file_name)['PPG'].to_numpy()
    test_data[i] = torch.tensor(signal_csv)
    test_label[i] = torch.tensor([AF_signal_label])

val_csv_path = '../../data/MixArtifacts/b30sec/val/'
val_df = pd.read_csv(
    '../../data/MixArtifacts/b30sec/val.csv', encoding='cp949')

val_data = torch.empty(len(val_df), 1, 3000)
val_label = torch.empty(len(val_data), 1)

for i in tqdm(range(len(val_df))):
    file_name = val_df.loc[i]['FileName']
    AF_signal_label = val_df.loc[i]['AF']
    signal_csv = pd.read_csv(val_csv_path+file_name)['PPG'].to_numpy()
    val_data[i] = torch.tensor(signal_csv)
    val_label[i] = torch.tensor([AF_signal_label])

train_dataset = CustomDataset(train_data, train_label)
test_dataset = CustomDataset(test_data, test_label)
val_dataset = CustomDataset(val_data, val_label)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


model = OmniScaleCNN(1, 1, 3000).to(device0)
accuracy = torchmetrics.Accuracy(task="binary", num_classes=1).to(device0)

criterion = nn.BCEWithLogitsLoss().to(device0)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

MIN_loss = 5000
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

for epoch in range(10000):

    train_count = 0
    running_loss = 0.0
    acc_loss = 0
    train_tqdm = tqdm(train_dataloader)
    for x, y in train_tqdm:
        model.train()
        y = y.to(device0).float()
        train_count += 1
        x = x.to(device0).float()
        optimizer.zero_grad()  # optimizer zero 로 초기화
        predict = model(x).to(device0)
        cost = criterion(predict, y)  # cost 구함
        acc = accuracy(predict, y)
        cost.backward()  # cost에 대한 backward 구함
        optimizer.step()
        running_loss += cost.item()
        acc_loss += acc
        train_tqdm.set_description(
            f"\repoch: {epoch+1}/{10000} train_loss : {running_loss/train_count:.4f} train_accuracy: {acc_loss/train_count:.4f}")
    train_loss_list.append((running_loss/train_count))
    train_acc_list.append((acc_loss/train_count).cpu().detach().numpy())
# validation
    model.eval()
    val_count = 0
    val_running_loss = 0.0
    val_acc_loss = 0
    validation_tqdm = tqdm(validation_dataloader)
    with torch.no_grad():
        for x, y in validation_tqdm:
            y = y.to(device0).float()
            val_count += 1
            x = x.to(device0).float()

            predict = model(x).to(device0)
            acc = accuracy(predict, y)
            cost = criterion(predict, y)
            val_running_loss += cost.item()
            val_acc_loss += acc
            validation_tqdm.set_description(
                f"\repoch: {epoch+1}/{10000} val_loss : {val_running_loss/val_count:.4f}  val_accuracy: {val_acc_loss/val_count:.4f}")

        val_loss_list.append((val_running_loss/val_count))
        val_acc_list.append((val_acc_loss/val_count).cpu().detach().numpy())

    if MIN_loss > (val_running_loss/val_count):
        torch.save(model.state_dict(
        ), '../../model/architecture_waveform/PPG_OmniScaleCNN_callback.pt')
        MIN_loss = (val_running_loss/val_count)
torch.save(model.state_dict(),
           '../../model/architecture_waveform/PPG_OmniScaleCNN.pt')
