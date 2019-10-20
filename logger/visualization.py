import importlib
from datetime import datetime
import wandb
import os
from pathlib import Path


class WandbWriter():
    def __init__(self, log_dir, logger, enabled=True):
        self.enabled = enabled is not False
        self.logger = logger

        if enabled:
            log_dir = str(log_dir)
            wandb.init(project=os.path.realpath(__file__).split('/')[-3], dir=log_dir)

        self.step = 0
        self.mode = ''

        self.timer = datetime.now()

    def set_step(self, step=None, mode='train', log=False):
        mode_change = self.mode != mode
        self.mode = mode
        if step is not None:
            self.step = step
        if step == 0 or mode_change:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            if log:
                self.log({'steps_per_sec': 1 / duration.total_seconds()})
            self.timer = datetime.now()

    def __getattr__(self, name):
        if self.enabled:
            if name == 'log':
                def wrapper(row=None, sync=False):
                    # add mode(train/valid) tag
                    row = {'{}/{}'.format(tag, self.mode): data for tag, data in row.items()}
                    wandb.log(row, step=self.step, sync=sync)
                return wrapper
            else:
                # default action for returning methods defined in this class, set_step() for instance.
                try:
                    attr = getattr(wandb, name)
                except AttributeError:
                    raise AttributeError("wandb has no attribute '{}'".format(name))
                return attr
        else:
            return EmptyWriter()

class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = Path(log_dir) / 'tensoboard'

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step=None, mode='train', log=False):
        old_mode = self.mode
        self.mode = mode
        if step is not None:
            self.step = step
        if step == 0 or old_mode != self.mode:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            if log:
                self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr

class EmptyWriter():
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return EmptyWriter()

    def __call__(self, *args, **kwargs):
        return EmptyWriter()

    def __setitem__(self, key, value):
        return EmptyWriter()

    def __getitem__(self, item):
        return EmptyWriter()
