import numpy as np
import os
import urllib.request
from scipy.io import loadmat
from datasets.timeseries import TimeSeries

SOURCE_URL = "https://github.com/locuslab/TCN/raw/master/TCN/poly_music/mdata/"
JSB_FILENAME = "JSB_Chorales.mat"
Nott_FILENAME = "Nottingham.mat"

class Music():
    def __init__(self, seq_len=100, data_dir="./data", dataset="JSB"):
        self.seq_len = seq_len
        if dataset == "JSB":
            self.filepath = os.path.join(data_dir, JSB_FILENAME)
        if dataset == "Nott":
            self.filepath = os.path.join(data_dir, Nott_FILENAME)
        self._maybe_download(data_dir, dataset)  # list of filepaths

        X_train, X_valid, X_test = self.load_series()

        X_train_t = self._change_to_seq_len(X_train, self.seq_len + 1)

        X_test_t = self._change_to_seq_len(X_test + X_valid, self.seq_len + 1)

        self.train = {'X': {'data': X_train_t[:, :self.seq_len , ...], 'axes': ('N', 'REC', 'F')}, 'y': {'data': X_train_t[:, 1:, ...], 'axes': ('N', 'REC', 'Fo')}}

        self.test = {'X': {'data': X_test_t[:, :self.seq_len, ...], 'axes': ('N', 'REC', 'F')}, 'y': {'data': X_test_t[:, 1:, ...], 'axes': ('N', 'REC', 'Fo')}}

    def load_series(self):
        data = loadmat(self.filepath)
        X_train = list(data['traindata'][0])
        X_valid = list(data['validdata'][0])
        X_test = list(data['testdata'][0])
        return X_train, X_valid, X_test

    def _change_to_seq_len(self, X, seq_len):
        X_padded = np.zeros((len(X), seq_len, X[0].shape[1]))

        for e, x in enumerate(X):
            if x.shape[0] >= seq_len:
                X_padded[e, :, :] = x[-1*seq_len:, :]
            else:
                X_padded[e, -1*x.shape[0]:, :] = x
        return X_padded

    def _maybe_download(self, work_directory, dataset):
        """
        This function downloads the stock data if its not already present

        Returns:
            Location of saved data

        """
        if (not os.path.exists(self.filepath)):
            print("data does not exist, downloading...")
            self._download_data(work_directory, dataset)

    def _download_data(self, work_directory, dataset):
        work_directory = os.path.abspath(work_directory)
        if not os.path.exists(work_directory):
            os.mkdir(work_directory)

        headers = {'User-Agent': 'Mozilla/5.0'}

        if dataset == "JSB":
            filename = JSB_FILENAME
        if dataset == "Nott":
            filename = Nott_FILENAME

        filepath = os.path.join(work_directory, filename)
        req = urllib.request.Request(SOURCE_URL + filename, headers=headers)
        data_handle = urllib.request.urlopen(req)
        with open(filepath, "wb") as fp:
            fp.write(data_handle.read())

        print('Successfully downloaded data to {}'.format(filepath))

        return filepath
