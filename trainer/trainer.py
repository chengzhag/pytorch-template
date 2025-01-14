import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from datetime import datetime, timedelta
from logger import *


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(len(data_loader)))

        self.train_metrics = MetricTracker()
        self.valid_metrics = MetricTracker()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        tic = datetime.now()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            do_log = batch_idx % self.log_step == 0

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, log=do_log)
            tag_scalar_dict = {'loss': loss.item()}
            for i, params in enumerate(self.optimizer.param_groups):
                tag_scalar_dict.update({f'lr{i}': params['lr']})
            tag_scalar_dict.update({met.__name__: met(output, target) for met in self.metric_ftns})
            self.train_metrics.update(tag_scalar_dict)
            if do_log:
                if isinstance(self.writer, TensorboardWriter):
                    for tag, scalar in tag_scalar_dict.items():
                        self.writer.add_scalar(tag, scalar)
                elif isinstance(self.writer, WandbWriter):
                    self.writer.log(tag_scalar_dict)

            toc = datetime.now()
            try:
                speed = 0.95 * speed + 0.05 * (toc - tic)
            except:
                speed = toc - tic
            eta = speed * (self.len_epoch - batch_idx - 1 + (self.epochs - epoch) * self.len_epoch)
            tic = toc

            if do_log:
                eta = timedelta(seconds=eta.seconds)
                self.logger.debug('Train Epoch: {} {}, Loss: {:.6f}, ETA: {}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    eta))
                if isinstance(self.writer, TensorboardWriter):
                    self.writer.add_scalar('ETAhours', eta.seconds/3600)
                elif isinstance(self.writer, WandbWriter):
                    self.writer.run.summary['ETA'] = str(eta)

            if batch_idx == self.len_epoch:
                break

        if isinstance(self.writer, TensorboardWriter):
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        elif isinstance(self.writer, WandbWriter):
            self.writer.log({'input': self.writer.Image(make_grid(data.cpu(), nrow=8, normalize=True))})
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            tic = datetime.now()
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                do_log = batch_idx % self.log_step == 0

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step(mode='valid', log=do_log)
                tag_scalar_dict = {'loss': loss.item()}
                tag_scalar_dict.update({met.__name__: met(output, target) for met in self.metric_ftns})
                self.valid_metrics.update(tag_scalar_dict)
                if do_log:
                    if isinstance(self.writer, TensorboardWriter):
                        for tag, scalar in tag_scalar_dict.items():
                            self.writer.add_scalar(tag, scalar)
                        self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                    elif isinstance(self.writer, WandbWriter):
                        self.writer.log(tag_scalar_dict)
                        self.writer.log({'input': self.writer.Image(make_grid(data.cpu(), nrow=8, normalize=True))})

                toc = datetime.now()
                try:
                    speed = 0.95 * speed + 0.05 * (toc - tic)
                except:
                    speed = toc - tic
                eta = speed * (len(self.valid_data_loader) - batch_idx)
                tic = toc

                if do_log:
                    current = batch_idx * self.valid_data_loader.batch_size
                    n_samples = len(self.valid_data_loader.sampler)
                    self.logger.debug('Test Epoch: {} [{}/{} ({:.0f}%)], ETA: {}'.format(
                        epoch,
                        current,
                        n_samples,
                        100.0 * current / n_samples,
                        timedelta(seconds=eta.seconds)
                    ))

        if isinstance(self.writer, TensorboardWriter):
            for tag, scalar in self.valid_metrics.result().items():
                self.writer.add_scalar(tag, scalar)
            # add histogram of valid results to the tensorboard
            for name, p in self.valid_metrics.to_dict().items():
                self.writer.add_histogram(name + '_h', p, bins='auto')

        elif isinstance(self.writer, WandbWriter):
            self.writer.log(self.valid_metrics.result())

            # add histogram of valid results to the tensorboard
            self.writer.log({name + '_h': p for name, p in self.valid_metrics.to_dict().items()})

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
