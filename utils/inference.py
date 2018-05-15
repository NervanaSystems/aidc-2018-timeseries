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
import numpy as np
# TODO figure out a way to move this to topologies.timeseries_model
def generate_sequence(data, time_points, eval_function, predict_seq,
                      batch_size, seq_len, feature_dim, seq_name=0):
    """
    Generates a sequence of length time_points, given ground truth data (gt_data)
    First seq_len points of gt_data is used as the seed
    Returns the generated sequence

    data: ground truth data
    time_points: number of steps to generate the data
    eval_function: forward prop function of the network
    predict_seq: True if network predicts sequences

    Start with first seq_len points in training data, take it as input (call S0)
    S0 = [x0, x1, ..., x(seq_len-1)]
    Given S0, generate next time point x_hat(seq_len), build S1
    S1 = [x1, x2, ..., x(seq_len-1), x_hat(seq_len)]
    Given S1, generate x_hat(seq_len+1)
    Continue generating for a total of time_points
    """
    # check if seq_name is in train or test
    if seq_name in data.test_seq_names:
        data_fold = data.test
        seq_indices = [e for e, n in enumerate(data.test_seq_names) if n == seq_name]
    elif seq_name in data.train_seq_names:
        data_fold = data.train
        seq_indices = [e for e, n in enumerate(data.train_seq_names) if n == seq_name]
    else:
        raise ValueError("Input sequence name {} not found".format(seq_name))

    # keep the last few indices in order to get most recent data
    seq_indices = seq_indices[-1 * (time_points + 1):]

    no_gen_time_points = time_points
    input_batch = np.zeros((batch_size, seq_len, feature_dim))

    input_batch[0] = data_fold['X']['data'][seq_indices[0]]
    gen_series = data_fold['X']['data'][seq_indices[0]]  # This will hold the generated series
    gt_series = data_fold['X']['data'][seq_indices[0]]  # This will hold the ground truth series

    output_dim = data_fold['y']['data'].shape[-1]
    for tp in range(no_gen_time_points):
        axx = dict(X=input_batch)
        # Get the prediction using seq_len past samples
        result = eval_function(axx)

        if(predict_seq is False):
            # result is of size (batch_size, output_dim)
            # We want the output of the first sample, so get it
            result = result[0, :]
        else:
            # result is of size (batch_size, seq_len, output_dim)
            # We want the last output of the first sample, so get it
            result = result[0, -1, :]
        # Now result is (output_dim,)
        # Reshape result to (1,output_dim)
        result = np.reshape(result, (1, output_dim))

        # Get the last (seq_len-1) samples in the past
        # cx is of shape (seq_len-1, output_dim)
        cx = input_batch[0][1:, :]

        # Append the new prediction to the past (seq_len-1) samples
        # Put the result into the first sample in the input batch
        input_batch[0] = np.concatenate((cx, result))

        # Append the current prediction to gen_series
        # This is to keep track of predictions, for plotting purposes only
        gen_series = np.concatenate((gen_series, result))

        # Find the ground truth for this prediction
        if(predict_seq is False):
            gt_outcome = np.copy(data_fold['X']['data'][seq_indices[tp + 1]][-1, :])
            # Reshape to (1, output_dim)
            gt_outcome = np.reshape(gt_outcome, (1, output_dim))
        else:
            # When predict_seq is given, input 'X' has non overlapping windows
            # X is of shape (no_samples, seq_len, 2)
            # Thus, find the right index for the ground truth output
            gt_outcome = data_fold['X']['data'][seq_indices[(tp + seq_len) // seq_len],
                                                 (tp + seq_len) % seq_len, :]
            # Reshape to (1, output_dim)
            gt_outcome = np.reshape(gt_outcome, (1, output_dim))

        # Append ground truth outcome to gt_series
        # This is to keep track of ground truth, for plotting purposes only
        gt_series = np.concatenate((gt_series, gt_outcome))

    return gen_series, gt_series
