from ngraph.frontends.neon import Sequential
from contextlib import closing
from ngraph.frontends.neon import Saver
from ngraph.frontends.neon.callbacks import tqdm
import ngraph.transformers as ngt
import numpy as np
import json
import os
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib not found")
from ngraph.frontends.neon.model import ResidualModule


class TimeseriesTrainer:
    """
    Class that adds methods for training, inference, model summary and tensorboard

    Arguments: TODO
        train_computation
        eval_computation
        pred_computation
        input_placeholders
        model_graph (list of graphs): each list element must have layers attribute, e.g. a Sequential
        tensorboard_dir (optional, path): if given, save tensorboard to this directory
    """
    def __init__(self, opt, train_computation, eval_computation, pred_computation, input_placeholders, model_graph,
                 tensorboard_dir=None):
        self.opt = opt
        self.train_computation = train_computation
        self.eval_computation = eval_computation
        self.pred_computation = pred_computation
        self.input_placeholders = input_placeholders
        self.layers = [layer for graph in model_graph for layer in graph.layers]
        self.transformer = ngt.make_transformer()  # initialize transformer
        self._init_tensorboard(tensorboard_dir, model_graph)

    def train(self, train_iterator, val_iterator, n_epochs=100, log_interval=100, save_plots=True, results_dir="./"):
        train_iterator.reset()
        val_iterator.reset()

        batch_size = train_iterator.batch_size
        num_iterations = np.floor((train_iterator.ndata * n_epochs * 1.)/batch_size).astype('int')
        n_train = train_iterator.ndata

        assert val_iterator.batch_size == batch_size

        # save model
        weight_saver = Saver()
        # train model

        self.train_function = self.transformer.add_computation(self.train_computation)
        self.eval_function = self.transformer.add_computation(self.eval_computation)
        self.pred_function = self.transformer.add_computation(self.pred_computation)

        # set up weight saver
        weight_saver.setup_save(transformer=self.transformer, computation=self.train_computation)

        # Progress bar
        tpbar = tqdm(unit="batches", ncols=100, total=num_iterations)
        tpbar_string = "Train Epoch:  {epoch} [ {num_examples_seen}/{n_train} ({percent_complete}%)] Train Loss {cost}"

        train_losses = []
        eval_losses = []

        # Iterating over the training set
        num_examples_seen = 0
        n_epoch = 1
        for step in range(num_iterations):
            data = next(train_iterator)
            feed_dict = {self.input_placeholders["X"]: data["X"], self.input_placeholders["y"]: data["y"]}

            # Mean batch cost
            output = self.train_function(feed_dict=feed_dict)
            train_loss = output[()].item()

            train_losses.append(train_loss)
            if self.tb is not None:
                self.tb.add_scalar("train_loss", train_loss, step=step)

            # Update progress bar
            tpbar.update(1)
            tpbar.set_description("Training {}".format(str(output[()])))

            num_examples_seen += batch_size
            # Every epoch print test set metrics
            if (step + 1) % log_interval == 0 and step > 0:

                # calculate metrics over test set
                avg_eval_loss = 0.0
                val_iterator.reset()
                for e, data_test in enumerate(val_iterator):
                    feed_dict_test = {self.input_placeholders["X"]: data_test["X"], self.input_placeholders["y"]: data_test["y"]}
                    eval_loss = self.eval_function(feed_dict=feed_dict_test)[0]
                    avg_eval_loss += eval_loss

                avg_eval_loss /= (e + 1)

                # save loss
                eval_losses.append(avg_eval_loss.item())
                if self.tb is not None:
                    self.tb.add_scalar("eval_loss", avg_eval_loss, step=step)

                # write to progress bar
                avg_train_cost = train_losses[-1 * log_interval:]
                avg_train_cost = np.mean(avg_train_cost)
                tqdm.write(tpbar_string.format(epoch=n_epoch, num_examples_seen=num_examples_seen, n_train=n_train, percent_complete=100.0 * num_examples_seen / n_train, cost=avg_train_cost))

                weight_saver.save(filename=results_dir + "/" + "model")

                # Writing to CSV
                logfile = os.path.join(results_dir, "logs")

                with open(logfile, 'w') as fp:
                    json.dump({'train_loss': train_losses, 'eval_loss': eval_losses}, fp)

                if save_plots:
                    # plot all entries in logfile
                    self.plot_scalars(logfile, results_dir)

            if num_examples_seen > n_train:
                num_examples_seen = num_examples_seen - n_train
                n_epoch += 1
                print("Test set: Average loss: {}".format(avg_eval_loss))

        print("\nTraining Completed")

    def predict(self, dataset, num_batches=None):
        """
            Runs a function over the dataset and accumulated the results.
            Instead of reducing the results, as ngraph's loops do, we stack them together.
            This allows us, for instance, to retain the predictions made on each test
            example.
            """
        dataset.reset()
        all_results = []

        for ee, data in enumerate(dataset):
            if num_batches is not None:
                if ee >= num_batches:
                    break
            results = self.pred_function(feed_dict={self.input_placeholders['X']: data['X']})[0]
            all_results.extend(list(results))
        all_results = np.stack(all_results, axis=0)
        all_results = all_results[:dataset.ndata]
        return all_results

    def predict_sequence(self):
        pass

    def callbacks(self):
        pass

    def plot_scalars(self, logfile, results_dir):
        with open(logfile, 'r') as fp:
            data = json.load(fp)

        for k in data:
            if isinstance(data[k], list):
                fig, ax = plt.subplots()
                plt.plot(data[k])
                plt.xlabel('Iteration')
                plt.ylabel('%s' % k)
                plt.title('%s ' % k)
                plt.savefig('%s' % os.path.join(results_dir, k + ".png"))
                plt.close()

    def summary(self):
        if self.layers is None:
            raise ValueError("Model layers not provided")
        total_num_vars = 0
        total_num_not_trainable = 0
        print("".join(100 * ["-"]))
        print("{: >20} {: >20} {: >20} {: >20} {: >20}".format("index", "name", "# trainable vars", "# not trainable vars", "output_shape"))
        print("".join(100*["-"]))
        for e, layer in enumerate(self.layers):
            temp_model = Sequential(self.layers[0:e+1])
            l_output = temp_model(self.input_placeholders['X'])
            num_vars, num_not_trainable = self._get_number_of_vars_in_layer(layer)
            if num_vars is not None:
                total_num_vars += num_vars
            if num_not_trainable is not None:
                total_num_not_trainable += num_not_trainable
            if 'name' in layer.__dict__:
                l_name = layer.name
            elif isinstance(layer, ResidualModule):
                l_name = 'ResidualModule'
            else:
                l_name = type(layer).__name__
            if 'axes' in dir(l_output):
                print("{: >20} {: >20} {: >20} {: >20} {: >20}".format(str(e), l_name, str(num_vars), str(num_not_trainable), str(l_output.axes)))
            else:
                print("{: >20} {: >20} {: >20} {: >20} {: >20}".format(str(e), l_name, str(num_vars), str(num_not_trainable), "Unknown"))

        print("".join(100 * ["-"]))
        print("Total number of trainable parameters: %d" % total_num_vars)
        print("Total number of non trainable parameters: %d" % total_num_not_trainable)
        print("".join(100 * ["-"]))
        print("Optimizer type {}".format(self.opt.name))
        print("Optimizer learning rate {}".format(self.opt.lrate.initial_value.item()))
        print("".join(100 * ["-"]))

    def _get_number_of_vars_in_layer(self, layer):
        num_vars = []
        num_not_trainable = []
        if 'variables' in dir(layer):
            for i in layer.variables.keys():
                if layer.variables[i].is_trainable:
                    num_vars.append(self._get_number_of_vars_in_tensor(layer.variables[i]))
                else:
                    num_not_trainable.append(self._get_number_of_vars_in_tensor(layer.variables[i]))
            return np.sum(num_vars).astype('int'), np.sum(num_not_trainable).astype('int')
        else:
            if isinstance(layer, ResidualModule):
                side_path_layers = [] if layer.side_path is None else layer.side_path.layers
                for l in layer.main_path.layers + side_path_layers:
                    num_vars_l, num_not_t_l = self._get_number_of_vars_in_layer(l)
                    num_vars.append(num_vars_l)
                    num_not_trainable.append(num_not_t_l)
                return np.sum(num_vars).astype('int'), np.sum(num_not_trainable).astype('int')
            else:
                return 0, 0

    def _get_number_of_vars_in_tensor(self, var):
        return np.prod(var.shape.full_lengths)

    def _init_tensorboard(self, tensorboard_dir, model_graph):
        self.tb = None
        if tensorboard_dir is not None:
            try:
                from ngraph.op_graph.tensorboard.tensorboard import TensorBoard
                self.tb = TensorBoard(tensorboard_dir)
                for graph in model_graph:
                    self.tb.add_graph(graph)
                # if not specifying run kwarg to TensorBoard or using add_run,
                # run attribute is autogenerated when add_graph is called
                print("Saving Tensorboard to {}/{}".format(tensorboard_dir, self.tb.run))
            except:
                print("Tensorboard not installed")
        else:
             print("no Tensorboard directory given, not using Tensorboard")
