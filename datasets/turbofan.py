"""
This class can be used to create a dataloader object for the turbofan engine degradation simulated data.
Data is a combination of two sources, and can be downloaded using the following links:
[1] https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan
[2] https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#phm08_challenge

For details regarding data generation, see:
[3] A. Saxena, K. Goebel, D. Simon, and N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

A list of papers that use the dataset are below:
[4] Heimes, F.O., Recurrent neural networks for remaining useful life estimation, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
[5] Tianyi Wang, Jianbo Yu,  Siegel, D.,  Lee, J., A similarity-based prognostics approach for Remaining Useful Life estimation of engineered systems, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
[6] Peel, L., Recurrent neural networks for remaining useful life estimation, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
[7] Ramasso, Emmanuel, and Abhinav Saxena. "Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets." International Journal of Prognostics and Health Management 5.2 (2014): 1-15.
"""
import os
import pandas as pd
import numpy as np
import urllib.request
import zipfile

CMAPSS_SOURCE_URL = 'https://ti.arc.nasa.gov/c/6/'
CHALLENGE_SOURCE_URL = 'https://ti.arc.nasa.gov/c/13/'
CMAPPS_FILENAME = 'CMAPSSData'
CHALLENGE_FILENAME = 'Challenge_Data'

class TurboFan:
    def __init__(self, data_dir="./data", T=100, skip=1, max_rul_predictable=200, scale=True, normalize=True, shuffle=True,
                 recurrent_axis_name='REC', feature_axis_name='F', label_axis_name='Fo'):
        """

        Args:
            data_dir: location of data dir
            T: int, sequence length
            skip: int, number of timepoints to skip before moving to the next window (similar to "stride" in convolutional filter)
            max_rul_predictable: int, value at which RUL value is capped
            scale: boolean, Scale each variable-length trajectory down to [0, 1] if true
            normalize: boolean, Scale each sample (after windowing) to zero mean unit variance if true
            shuffle: boolean, shuffle training data if true
        """
        self.n_sensors = 21
        self.n_operating_modes = 3
        self.data_dir = data_dir
        self.T = T
        self.skip = skip
        self.max_rul_predictable = max_rul_predictable
        self.CMAPSSDir = os.path.join(data_dir, "CMAPSSData")
        self.Challenge_Data = os.path.join(data_dir, "Challenge_Data")
        self.filepath = self._maybe_download(data_dir) # list of filepaths

        print("Loading data")
        self.train_trajectories, self.val_trajectories, self.val_rul, self.test_trajectories = self.load_series()
        if scale:
            self.train_trajectories = self.scale_data(self.train_trajectories)
            self.val_trajectories = self.scale_data(self.val_trajectories)
            self.test_trajectories = self.scale_data(self.test_trajectories)

        self.n_features = self.train_trajectories[0].shape[1]

        print("Creating sliding window data")
        X_train, y_train = self.sliding_window_rul(self.train_trajectories, skip=skip)

        if normalize:
            X_train = self.normalize_data(X_train)
        if shuffle:
            X_train, y_train = self.shuffle_data(X_train, y_train)

        X_train_prev = np.roll(X_train, shift=1, axis=1)

        X_val, y_val = self.sliding_window_rul(self.val_trajectories, rul=self.val_rul, augment_test_data=False)
        if normalize:
            X_val = self.normalize_data(X_val)
        X_val_prev = np.roll(X_val, shift=1, axis=1)

        self.train = {'X': {'data': X_train, 'axes': ('N', recurrent_axis_name, feature_axis_name)},
                      'X_prev': {'data': X_train_prev, 'axes': ('N', recurrent_axis_name, feature_axis_name)},
                      'y': {'data': y_train, 'axes': ('N', label_axis_name)}}

        self.test = {'X': {'data': X_val, 'axes': ('N', recurrent_axis_name, feature_axis_name)},
                     'X_prev': {'data': X_val_prev, 'axes': ('N', recurrent_axis_name, feature_axis_name)},
                     'y': {'data': y_val, 'axes': ('N', label_axis_name)}}

        print("Done. Number of samples in train: {}, number of samples in test: {}".format(len(X_train), len(X_val)))


    def load_series(self):
        train_trajectories = []
        val_trajectories = []
        val_rul = []
        test_trajectories = []

        # CMAPSS data
        for f in ["FD00" + str(i+1) for i in range(4)]:
            full_path = os.path.join(self.CMAPSSDir, "train_" + f + ".txt")
            train_trajectories += self.load_data_from_file(full_path)

            full_path = os.path.join(self.CMAPSSDir, "test_" + f + ".txt")
            val_trajectories += self.load_data_from_file(full_path)

            full_path = os.path.join(self.CMAPSSDir, "RUL_" + f + ".txt") # this RUL corresponds to the val trajectory
            fp = open(full_path, "r")
            val_rul += [int(line.strip("\n")) for line in fp]

        assert len(val_trajectories) == len(val_rul)

        # Challenge data
        full_path = os.path.join(self.Challenge_Data, "train.txt")
        train_trajectories += self.load_data_from_file(full_path)

        full_path = os.path.join(self.Challenge_Data, "test.txt") # this data does not have RUL
        test_trajectories += self.load_data_from_file(full_path)

        full_path = os.path.join(self.Challenge_Data, "final_test.txt") # this data does not have RUL
        test_trajectories += self.load_data_from_file(full_path)

        return train_trajectories, val_trajectories, val_rul, test_trajectories

    def sliding_window_rul(self, trajectories, skip=1, rul=None, augment_test_data=False):
        """
        Given a set of trajectories, split into equal sized windows and corresponding rul values
        If rul is not provided, the end of the trajectory is assumed to be the time of failure
        Args:
            trajectories: List of numpy arrays, elements have variable dim-0 (time) and same number of attributes = (self.n_features)
            rul: remaining useful life

        Returns:
            X, y: X is numpy array of shape (N, self.T, self.n_features), y is numpy array of size (N,)

        """
        X = []
        y = []
        for ii, traj in enumerate(trajectories):
            # backfill all trajectories that are smaller than desired trajectory
            if traj.shape[0] < self.T * skip:
                padded_traj = np.zeros((self.T * skip, traj.shape[1]))
                padded_traj[-1*traj.shape[0]:, :] = traj
                padded_traj[0:self.T * skip - traj.shape[0], :] = traj[0, :]
                traj = padded_traj

            assert np.any(np.isnan(traj)) == False

            if rul is None or (rul is not None and augment_test_data):
                shape = [int(np.ceil((traj.shape[0] - self.T + 1) * 1.0/ skip)), self.T, traj.shape[-1]]
                strides = [traj.strides[0] * skip, traj.strides[0], traj.strides[-1]]
                strided_a = np.lib.stride_tricks.as_strided(traj, shape=shape, strides=strides, writeable=False)
                X.append(strided_a)
                if rul is None:
                    y.append(traj.shape[0] - self.T - np.arange(0, traj.shape[0] - self.T + 1, skip)[:, np.newaxis])
                else:
                    y.append(rul[ii] + traj.shape[0] - self.T - np.arange(0, traj.shape[0] - self.T + 1, skip)[:, np.newaxis])
            else:
                X.append(traj[-1*self.T:, :][np.newaxis, :, :])
                y.append(np.array([rul[ii]])[:, np.newaxis])

        X = np.vstack(X)
        y = np.vstack(y)
        y[y > self.max_rul_predictable] = self.max_rul_predictable

        assert X.shape[0] == y.shape[0]

        assert np.all(y >= 0) == True

        return X, y


    def load_data_from_file(self, f):
        df = pd.read_csv(f, sep=' ', header=None, index_col=False).fillna(method='bfill')
        df = df.dropna(axis='columns', how='all')
        assert df.shape[1] == self.n_sensors + self.n_operating_modes + 2
        df.columns = ["trajectory_id", "t"] + ["setting_" + str(i + 1) for i in range(self.n_operating_modes)] + ["sensor_" + str(i + 1) for i in range(self.n_sensors)]
        grouped = df.groupby("trajectory_id")
        trajectories = []
        for traj_id, traj in grouped:
            trajectories.append(traj[["setting_" + str(i + 1) for i in range(self.n_operating_modes)] + ["sensor_" + str(i + 1) for i in range(self.n_sensors)]].as_matrix())
        return trajectories

    def _maybe_download(self, work_directory):
        """
        This function downloads the stock data if its not already present

        Returns:
            Location of saved data

        """
        if (not os.path.exists(self.CMAPSSDir)) or len(os.listdir(self.CMAPSSDir)) == 0:
            print("CMAPSS data does not exist, downloading...")
            self._download_data(work_directory, "CMAPSS")

        if (not os.path.exists(self.Challenge_Data)) or len(os.listdir(self.Challenge_Data)) == 0:
            print("Challenge data does not exist, downloading...")
            self._download_data(work_directory, "Challenge")

    def _download_data(self, work_directory, dataset):
        work_directory = os.path.abspath(work_directory)
        if not os.path.exists(work_directory):
            os.mkdir(work_directory)

        headers = {'User-Agent': 'Mozilla/5.0'}

        if dataset == "CMAPSS":
            SOURCE_URL = CMAPSS_SOURCE_URL
            filename = CMAPPS_FILENAME
        if dataset == "Challenge":
            SOURCE_URL = CHALLENGE_SOURCE_URL
            filename = CHALLENGE_FILENAME

        filepath = os.path.join(work_directory, filename + ".zip")
        req = urllib.request.Request(SOURCE_URL, headers=headers)
        data_handle = urllib.request.urlopen(req)
        with open(filepath, "wb") as fp:
            fp.write(data_handle.read())

        print('Successfully downloaded data to {}'.format(filepath))

        unzip_dir = os.path.join(work_directory, filename)
        if not os.path.exists(unzip_dir):
            os.mkdir(unzip_dir)
        fp = zipfile.ZipFile(filepath, 'r')
        fp.extractall(unzip_dir)
        fp.close()
        print('Successfully unzipped data to {}'.format(unzip_dir))

        return filepath

    def normalize_data(self, X):
        """
        Normalize each sample - for sensors, zero mean and std normalization, for op points, scale only
        Args:
            X: samples

        Returns:
            Normalized sample matrix

        """
        # cols to normalize
        cols_to_normalize = list(set(range(0, self.n_sensors + self.n_operating_modes)) - set([0, 1, 2]))
        X_mean = np.mean(X[:, :, cols_to_normalize], axis=1)
        X[:, :, cols_to_normalize] = X[:, :, cols_to_normalize] - X_mean[:, np.newaxis, :]
        X_std = np.std(X[:, :, cols_to_normalize], axis=1)
        X_std[X_std <= np.finfo(np.float64).eps] = 1
        X[:, :, cols_to_normalize] = X[:, :, cols_to_normalize]/X_std[:, np.newaxis, :]

        # normalize operating cond separately
        X[:, :, 0] = X[:, :, 0] / 100
        X[:, :, 1] = X[:, :, 1] / 100
        X[:, :, 2] = X[:, :, 2] / 100
        return X

    def scale_data(self, trajectories):
        """
        Scale each trajectory by its max value
        Args:
            trajectories: list of trajectories

        Returns:
            list of scaled trajectories

        """
        traj_scaled = []
        for traj in trajectories:
            traj = traj/np.max(traj, axis=0)
            traj_scaled.append(traj)

        return traj_scaled

    def shuffle_data(self, X, y):
        assert len(X) == len(y)
        shuffle_ind = np.random.permutation(len(X))
        X = X[shuffle_ind, ...]
        y = y[shuffle_ind, ...]
        return X, y

    def plot_sample(self, results_dir, trajectory_id=1):
        """
        Plots the trajectory of all sensors and operating modes from a chosen sample
        Args:
            results_dir: Directory to write plots to
            trajectory_id: index of trajectory within dataset

        Returns:
            None
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib not found")

        all_traj_ids = [trajectory_id]

        ncols = int(np.ceil((self.n_sensors + self.n_operating_modes)*1.0//3))

        fig, ax = plt.subplots(ncols, 3)
        fig.set_figheight(20)
        fig.set_figwidth(10)
        for ii in all_traj_ids:
            for jj in range(self.n_sensors):
                plt.subplot(ncols, 3, jj + 1)
                plt.plot(self.train_trajectories[ii][:, jj + self.n_operating_modes])
                plt.title('Sensor %d' % (jj + 1))

            for jj in range(self.n_operating_modes):
                plt.subplot(ncols, 3, jj + self.n_sensors + 1)
                plt.plot(self.train_trajectories[ii][:, jj])
                plt.title('Operating mode %d' % (jj + 1))

        plt.tight_layout()
        plt.savefig('%s' % os.path.join(results_dir, "trajectories.png"))
