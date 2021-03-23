from multiprocessing.dummy import freeze_support

import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class Nasdaq100padding(Dataset):
    """Nasdaq100padding dataset."""

    def __init__(self, normalize=None, partition="train", window=10, time_to_predict=10):
        self.df = pd.read_csv("filelists/Nasdaq_100/nasdaq100_padding.csv")
        # self.df = pd.read_csv("nasdaq100_padding.csv")
        self.partition = partition
        self.window = window
        self.time_to_predict = time_to_predict

        if normalize:
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(self.df)
            self.df = pd.DataFrame(x_scaled, columns=self.df.columns)
        x_train, x_test = train_test_split(self.df, test_size=0.2, random_state=42, shuffle=False)
        self.df_test = pd.DataFrame(x_test, columns=self.df.columns).reset_index(drop=True)
        self.df_train = pd.DataFrame(x_train, columns=self.df.columns).reset_index(drop=True)

    def __len__(self):
        if self.partition == "train":
            return len(self.df_train) - self.window - self.time_to_predict
        if self.partition == "test":
            return len(self.df_test) - self.window - self.time_to_predict
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        begin = idx
        end_of_x = idx + self.window
        end_of_y = idx + self.window + self.time_to_predict
        if self.partition == "train":
            return torch.FloatTensor(list(range(begin, end_of_x))), self.df_train.loc[begin:end_of_x - 1].values
        if self.partition == "test":
            return torch.FloatTensor(list(range(begin, end_of_x))), self.df_test.loc[begin:end_of_x - 1].values
        else:
            raise NotImplementedError


# example
if __name__ == '__main__':
    freeze_support()
    nasdaq100padding = Nasdaq100padding(True, "test", 10, 10)
    dataset_loader = torch.utils.data.DataLoader(nasdaq100padding,
                                                 batch_size=4, shuffle=True)

    x, y = next(iter(dataset_loader))
    print(x.reshape(4,10,1))
    print(y[:,:,0].shape)
