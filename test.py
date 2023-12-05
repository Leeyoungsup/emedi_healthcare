import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
from tqdm.auto import tqdm
import torchmetrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from copy import copy
from sklearn.metrics import confusion_matrix
device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1


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
    def __init__(self, data, label, data_list, transform=None, target_transform=None):
        self.PPG_data = data
        self.PPG_label = label
        self.PPG_data_path = data_list

    def __len__(self):
        return len(self.PPG_data)

    def __getitem__(self, idx):
        signal_tensor = self.PPG_data[idx]
        AF_signal_label = self.PPG_label[idx]
        AF_data_path = self.PPG_data_path[idx]
        return signal_tensor, AF_signal_label, AF_data_path
# model


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=50,
                               kernel_size=3, padding='same', padding_mode='replicate')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(0.5)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=50, out_channels=50,
                               kernel_size=3, padding='same', padding_mode='replicate')
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout1d(0.5)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=5)
        self.fc1 = nn.Linear(1500, 200, bias=False)
        self.fc2 = nn.Linear(200, 1, bias=False)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Data Load
test_csv_path = './data'
test_data_list = glob(test_csv_path+'/**/*.csv')
test_data = torch.empty(len(test_data_list), 1, 750)
test_label = torch.empty(len(test_data_list), 1)
for i in tqdm(range(len(test_data_list))):
    signal_csv = pd.read_csv(test_data_list[i])['PPG'].to_numpy()
    test_data_list[i].find('positive')
    AF_signal_label = 0
    if test_data_list[i].find('positive') != -1:
        AF_signal_label = 1
    else:
        AF_signal_label = 0

    test_data[i] = torch.tensor(resized(signal_csv, 750))
    test_label[i] = torch.tensor([AF_signal_label])
test_dataset = CustomDataset(test_data, test_label, test_data_list)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# model_load
accuracy = torchmetrics.Accuracy(task="binary", num_classes=1).to(device0)
model = CNN1D().to(device0)
criterion = nn.BCEWithLogitsLoss().to(device0)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
model.eval()
model.load_state_dict(torch.load(
    './model/CNN_1D.pt'))

# Predict
m = nn.Sigmoid()
y_30_25Hz = torch.empty((0, 1))
predict_30_25Hz = torch.empty((0, 1))
file_list = []
with torch.no_grad():
    for x, y, path in test_dataloader:
        y = y.to(device0).float()
        x = x.to(device0).float()
        y_30_25Hz = torch.cat([y_30_25Hz, y.cpu()])
        predict = model(x).to(device0)
        predict_30_25Hz = torch.cat([predict_30_25Hz, m(predict).cpu()])
        acc = accuracy(predict, y)
        cost = criterion(predict, y)
        file_list.append(path[0])
y_30_25Hz_score = roc_auc_score(y_30_25Hz, predict_30_25Hz)
print(f'30Sec_25Hz (AUC={y_30_25Hz_score :.4f})')

a = np.linspace(0.000, 1.000, 1001)
report1 = 0
tresh_hold = 0.000
for i in a:
    t_prob = np.where(predict_30_25Hz > i, 1, 0)
    report = f1_score(y_30_25Hz, t_prob)
    if report1 <= report:
        tresh_hold = copy(i)
        report1 = copy(report)
classes = ['Non-AF', 'AF']
t_prob = np.where(predict_30_25Hz > tresh_hold, 1, 0)
report = classification_report(y_30_25Hz, t_prob, target_names=classes)
pd.DataFrame({'FilePath': file_list, 'AF_GT': np.array(y_30_25Hz)[:, 0],
             'AF_pred': t_prob[:, 0]}).to_csv('./data/predict.csv', index=False)
print(f"Threshholds= {tresh_hold} f1-score={report1}")
print(report)
cm = confusion_matrix(y_30_25Hz, t_prob)
