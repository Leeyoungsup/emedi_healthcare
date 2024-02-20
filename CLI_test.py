import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
import torchmetrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from copy import copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from tsai.all import *
import os
import argparse
from urllib.parse import urlparse
from io import StringIO


device0 = torch.device("cpu")


class CustomDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.PPG_data = data

    def __len__(self):
        return len(self.PPG_data)

    def __getitem__(self, idx):

        signal_tensor = self.PPG_data[idx]

        return signal_tensor


def test(model, dataloader):
    validation_tqdm = tqdm(dataloader)
    model.eval()
    val_count = 0
    predict_list = []
    with torch.no_grad():
        for x in validation_tqdm:
            val_count += 1
            x = x.to(device0).float()
            predict = model(x).to(device0)
            predict_list.append(F.sigmoid(predict).item())
    return predict_list


def interpolate_signal(signal, old_fs, new_fs):
    # Interpolate a signal from old_fs to new_fs using linear interpolation.
    # Create a time array for the original signal
    old_time = np.linspace(0, len(signal) / old_fs,
                           len(signal), endpoint=False)

    # Create a time array for the resampled signal
    new_time = np.linspace(0, len(signal) / old_fs,
                           int(len(signal) * new_fs / old_fs), endpoint=False)

    # Use numpy's interp function to perform the linear interpolation
    new_signal = np.interp(new_time, old_time, signal)

    return new_signal


def main():
    parser = argparse.ArgumentParser(description="A simple Python CLI.")

    # Add command-line arguments
    parser.add_argument('--csv', help='Input csv file path', default=None)
    parser.add_argument('--url', help='Input url', default=None)
    parser.add_argument('--json', help='Input json file path', default=None)
    args = parser.parse_args()

    batch_size = 1

    if args.csv == None and args.url == None and args.json == None:
        print('csv파일이나 url, json을 입력해주세요. ex) python CLI_test.py --csv ./data/test/Type_1_22.csv')
        return False

    if args.csv != None and args.url != None and args.json != None:
        print('csv파일 또는 url, json을 하나만 입력해주세요.')
        return False
    if args.url != None:
        url = args.url
        response = requests.get(url)
        if response.status_code == 200:
            # Assuming the CSV content is UTF-8 encoded
            csv_data = response.content.decode('utf-8')
            parsed_url = urlparse(url)
            filename = parsed_url.path.split('/')[-1]
            # Use pandas to read the CSV data into a DataFrame
            df = pd.read_csv(StringIO(csv_data))
            df.rename(columns={'PPG': 'AF', 'Type_1': 'PPG'}, inplace=True)
            # Now 'df' is a pandas DataFrame containing the CSV data
            test_data = torch.empty(len(df)//2500+1, 1, 3000)
            for i in range(len(df)//2500):
                if len(df)//2500-1 == i:
                    signal_csv = df.loc[i*2500-499:(i+1)*2500]['PPG']
                    test_data[i] = torch.tensor(signal_csv.to_numpy())
                    signal_csv = df.loc[len(df)-3000:]['PPG']
                    test_data[i+1] = torch.tensor(signal_csv.to_numpy())
                else:
                    signal_csv = df.loc[i*2500:(i+1)*2500+499]['PPG']
                    test_data[i] = torch.tensor(signal_csv.to_numpy())
        else:
            print(
                f"Failed to download the file. Status code: {response.status_code}")
            return False

    if args.csv != None:
        try:
            df = pd.read_csv(args.csv, encoding='cp949')
            df.rename(columns={'PPG': 'AF', 'Type_1': 'PPG'}, inplace=True)
            filename = args.csv.split('/')[-1]
        except:
            print('파일이 없거나 존재하지 않습니다.')
            return False

        # Now 'df' is a pandas DataFrame containing the CSV data
        test_data = torch.empty(len(df)//2500+1, 1, 3000)
        for i in range(len(df)//2500):
            if len(df)//2500-1 == i:
                signal_csv = df.loc[i*2500-499:(i+1)*2500]['PPG']
                test_data[i] = torch.tensor(signal_csv.to_numpy())
                signal_csv = df.loc[len(df)-3000:]['PPG']
                test_data[i+1] = torch.tensor(signal_csv.to_numpy())
            else:
                signal_csv = df.loc[i*2500:(i+1)*2500+499]['PPG']
                test_data[i] = torch.tensor(signal_csv.to_numpy())

    if args.json != None:
        json_path = args.json
        filename = json_path.split('/')[-1]
        with open(json_path, 'r') as f:

            json_data = json.load(f)

        # Now 'ppg_green' is a pandas DataFrame containing the CSV data
        test_data = torch.empty(len(json_data), 1, 3000)
        for i in range(len(json_data)):
            ppg_green = json_data[i]['ppgGreen'].split(',')
            ppg_green = np.array(ppg_green, dtype=np.float32)
            ppg_green = interpolate_signal(ppg_green, 25, 100)
            if len(ppg_green) >= 3000:
                ppg_green = ppg_green[len(
                    ppg_green)//2-1500:len(ppg_green)//2+1500]
            else:
                ppg_green = np.concatenate(
                    (ppg_green, np.zeros(3000-len(ppg_green))), axis=0)
            test_data[i, 0, :] = torch.tensor(ppg_green)

    test_dataset = CustomDataset(test_data)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = GRU_FCN(1, 1, 3000).to(device0)
    model.load_state_dict(torch.load(
        '../../model/architecture_waveform/PPG_GRU_FCN_callback.pt', map_location=device0), strict=False)
    predict_list = test(model, test_dataloader)
    predict_array = np.array(predict_list).round(3)
    path_list = []
    print(predict_list)
    for i in range(len(predict_array)):
        path_list.append(filename+'_'+str(i))
    pd.DataFrame({'FilePath': np.array(path_list), 'AF_pred': predict_array}).to_csv(
        './data/'+os.path.splitext(filename)[0]+'predict.csv', index=False)

    with open('./data/'+os.path.splitext(filename)[0]+'predict.csv', 'r', newline='') as infile:
        reader = csv.reader(infile)
        data = list(reader)
    # 데이터 수정 (예: 첫 번째 행의 첫 번째 열 값을 변경)
    data.append([''])
    data.append(['min', 'mean', 'max'])
    data.append(
        [predict_array.min(), predict_array.mean(), predict_array.max()])

    # 수정된 데이터를 CSV 파일에 쓰기
    with open('./data/'+os.path.splitext(filename)[0]+'predict.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)


if __name__ == "__main__":
    main()
