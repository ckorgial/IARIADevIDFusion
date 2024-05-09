import os
import csv
import shutil
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def check_run_folder(exp_folder):
    run = 1
    run_folder = os.path.join(exp_folder, 'run{}'.format(run))
    if not os.path.exists(run_folder):
        Path(run_folder).mkdir(parents=True, exist_ok=True)

        # os.makedirs(run_folder)
        print("Path {} created".format(run_folder))
        return run_folder

    while os.path.exists(run_folder):
        run += 1
        run_folder = os.path.join(exp_folder, 'run{}'.format(run))
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    # os.makedirs(run_folder)
    print("Path {} created".format(run_folder))
    return run_folder


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def save_checkpoint(state, is_best, checkpoint_dir):
    filename = os.path.join(checkpoint_dir, 'model.ckpt')
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist")
        os.mkdir(checkpoint_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, "model_best.ckpt"))


def save_cm_fig(cm, classes, normalize, title, save_dir, figsize):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    else:
        cm = cm.astype('int')
    import seaborn as sns

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, cmap='Blues',
                fmt='.1%' if normalize else "d",
                annot_kws={"fontsize": 8},
                xticklabels=classes,
                yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{save_dir}/cm_{title}.jpg', bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{save_dir}/cm_{title}.eps', bbox_inches='tight', pad_inches=0, format='eps')
    plt.close()
