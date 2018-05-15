#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import division
import numpy as np


class TimeSeries(object):
    """
    Class that generates training and testing data from a time-series, for forecasting objectives.
    """

    def __init__(self, train_ratio=0.8, seq_len=30,
                 predict_seq=True, look_ahead=1):
        """
        Arguments:
            train_ratio (float, optional): percentage of the function to be used for training
            seq_len (int, optional): length of the sequence for each sample
            predict_seq (boolean, optional):
                False : Inputs - X[no_samples, seq_len, no_input_features]
                        Labels - y[no_samples, no_output_features]
                True : Inputs - X[no_samples, seq_len, no_input_features]
                       Labels - y[no_samples, seq_len, no_output_features]
            look_ahead (int, optional): How far ahead the predicted sequence starts from the input seq
                        Set to 1 to start predicting from next time point onwards
                        Only used when predict_seq is False
        """

        self.data, self.names = self.load_series() # data could be 2-D (single series) or 3-D (multiple series), in case of multiple series, each series could have different sequence lengths

        self.series = self.data

        if (predict_seq is False):
            # X will be (no_samples, time_steps, feature_dim)
            X, self.seq_names = self.rolling_window(a=self.data, seq_len=seq_len+1, seq_names=self.names) # add one to the sequence length to get target in the same call

            X, self.seq_names = self._remove_zero_batches(X, self.seq_names)

            # Get test samples
            test_samples = int(round((1 - train_ratio) * X.shape[0]))
            train_samples = X.shape[0] - test_samples

            self.train = {'X': {'data': X[:train_samples, :seq_len, ...], 'axes': ('N', 'REC', 'F')},
                          'y': {'data': X[:train_samples, seq_len, ...],
                                'axes': ('N', 'Fo')}}
            self.train_seq_names = self.seq_names[:train_samples]

            self.test = {'X': {'data': X[train_samples:, :seq_len, ...], 'axes': ('N', 'REC', 'F')},
                         'y': {'data': X[train_samples:, seq_len, ...],
                               'axes': ('N', 'Fo')}}
            self.test_seq_names = self.seq_names[train_samples:]

        else:
            X, y, self.seq_names = self.non_overlapping_window(self.data, seq_len=seq_len, look_ahead=look_ahead, seq_names=self.names)

            X, self.seq_names = self._remove_zero_batches(X, self.seq_names)


            test_samples = int(round((1 - train_ratio) * X.shape[0])) #TODO split strategies other than time
            train_samples = X.shape[0] - test_samples

            self.train = {'X': {'data': X[:train_samples], 'axes': ('N', 'REC', 'F')},
                          'y': {'data': y[:train_samples],
                                'axes': ('N', 'REC', 'Fo')}}
            self.train_seq_names = self.seq_names[:train_samples]

            self.test = {'X': {'data': X[train_samples:], 'axes': ('N', 'REC', 'F')},
                         'y': {'data': y[train_samples:], 'axes': ('N', 'REC', 'Fo')}}
            self.test_seq_names = self.seq_names[train_samples:]

    def load_series(self):
        raise NotImplementedError

    def convert_to_trend(self):
        # TODO work on trend targets: higher/lower/nochange from prev day's value
        raise NotImplementedError

    @staticmethod
    def rolling_window(a=None, seq_len=None, seq_names=None):
        """
        Convert sequence a into time-lagged vectors
        a           : (time_steps, feature_dim) or list of variable length series with same number of features
        seq_len     : length of sequence used for prediction
        returns  (n*(time_steps - seq_len + 1), seq_len, feature_dim)  array
        """
        if not isinstance(a, list):
            a = [a]

        all_windows = []
        all_seq_names = []
        for i, a_i in enumerate(a):
            if a_i.shape[0] < seq_len:
                continue
            shape = [a_i.shape[0] - seq_len + 1, seq_len, a_i.shape[-1]]
            strides = [a_i.strides[0], a_i.strides[0], a_i.strides[-1]]
            strided_a =  np.lib.stride_tricks.as_strided(a_i, shape=shape, strides=strides, writeable=False)
            all_windows.append(strided_a)
            if seq_names is None:
                all_seq_names += (a_i.shape[0] - seq_len + 1) * [i]
            else:
                all_seq_names += (a_i.shape[0] - seq_len + 1) * [seq_names[i]]

        return np.vstack(all_windows), all_seq_names

    def non_overlapping_window(self, a=None, seq_len=None, look_ahead=None, seq_names=None):
        """
        Convert sequence into (data, target) pairs with same sequence length in data and target and non-overlapping windows in data
        Args:
            a: input series
            seq_len: sequence length of data and target
            look_ahead: number of time points by which target is ahead of the data
            seq_labels: list containing name of each sequence
        Returns:

        """
        if not isinstance(a, list):
            a = [a]

        X = []
        y = []
        all_seq_names = []
        for i, a_i in enumerate(a):
            ntimepoints = ((a_i.shape[0] - look_ahead) // seq_len) * seq_len
            X_i = a_i[:ntimepoints, :]
            y_i = a_i[look_ahead:look_ahead + ntimepoints, :]

            # Reshape X and y
            nseq = ntimepoints // seq_len
            X_i = np.reshape(X_i, (nseq, seq_len, -1))
            y_i = np.reshape(y_i, (nseq, seq_len, -1))

            X.append(X_i)
            y.append(y_i)

            if seq_names is None:
                all_seq_names += nseq * [i]
            else:
                all_seq_names += nseq * [seq_names[i]]

        return np.vstack(X), np.vstack(y), all_seq_names

    def _remove_zero_batches(self, X, seq_labels):
        """
        This function removes batches which have all zero sequences, otherwise batch norm returns nan values
        Args:
            X: batch of data (batches, time, features)

        Returns:
            X with all zero batches removed

        """
        batch_sum = np.sum(np.sum(np.abs(X), axis=-1), axis=-1)
        non_zero_ind = np.where(batch_sum > np.finfo(np.float64).eps)[0]

        return X[non_zero_ind, :, :], [seq_labels[i] for i in non_zero_ind]

    def plot_sample(self, *args, **kwargs):
        raise NotImplementedError


    def normalize_data(self):
        raise NotImplementedError


    def _maybe_download(self, work_directory):
        raise NotImplementedError
