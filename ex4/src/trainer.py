import torch as t
import numpy as np
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()
        # -propagate through the network
        y_pred = self._model(x)
        # -calculate the loss
        loss = self._crit(y_pred, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()

    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        pred = self._model(x)
        loss = self._crit(pred, y)
        # return the loss and the predictions
        return loss.item(), pred

    def train_epoch(self):
        # set training mode
        self._model.train(mode=True)
        loss_total = 0
        count = 0
        # iterate through the training set
        for samples in self._train_dl:
            # imgdata, torch.tensor(two_label)
            img = samples[0]
            label = samples[1]
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda is True:
                img = img.cuda()
                label = label.cuda()
            # perform a training step
            loss_this_sample = self.train_step(img, label)
            loss_total += loss_this_sample
            count += 1
        # calculate the average loss for the epoch and return it
        avg_loss = loss_total / count
        return avg_loss

    def val_test(self):
        # set eval mode
        self._model.eval()
        label_save = []
        pred_save = []
        score_save = []
        loss_total = 0
        count = 0
        # disable gradient computation
        with t.no_grad:
            # iterate through the validation set
            for samples in self._val_test_dl:
                img = samples[0]
                label = samples[1]
                # transfer the batch to the gpu if given
                if self._cuda is True:
                    img = img.cuda()
                    label = label.cuda()
                # perform a validation step
                loss, pred = self.val_test_step(img, label)
                if self._cuda is True:
                    label_save.append(label.cpu().numpy())  # numpy cant load cuda tensor, so first be cpu tensor
                    pred_numpy = pred.cpu().numpy()
                    pred_numpy = np.round(pred_numpy)
                    pred_save.append(pred_numpy)
                score_batch = f1_score(label, pred_numpy, average='macro')
                score_save.append(score_batch)
                loss_total += loss
                count += 1
            # save the predictions and the labels for each batch
            # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
            avg_loss = loss_total / count
            avg_score = sum(score_save) / count
            print("Avg validation loss", avg_loss)
            print("Avg validation f1 score", avg_score)
        # return the loss and print the calculated metrics
        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch

        train_loss = []
        val_loss = []
        epoch_counter = 0
        len_without_improvement = 0
        while True:

            # stop by epoch number
            if epoch_counter == epochs:
                break
            epoch_counter += 1
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss_epoch = self.train_epoch()
            val_loss_epoch = self.val_test()
            # append the losses to the respective lists
            train_loss.append(train_loss_epoch)
            val_loss_last = val_loss[-1]
            val_loss.append(val_loss_epoch)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)

            if val_loss_epoch < val_loss_last:
                self.save_checkpoint(epoch_counter)
                len_without_improvement = 0
            else:
                len_without_improvement += 1
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if len_without_improvement >= self._early_stopping_patience:
                break
            # return the losses for both training and validation
            return train_loss, val_loss
