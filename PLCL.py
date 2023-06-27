import torch
import os
import time
import shutil
import argparse
import numpy as np
import copy
import torchvision
import time
from PIL import Image
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import datasets, transforms
import torch.nn.functional as F
from collections import OrderedDict as OD
import tensorboard_logger as tb_logger
from utils.utils_algo import *
from utils.randaugment import *
from model import PiCO
from resnet import *
import torch.backends.cudnn as cudnn
from utils.utils_loss import partial_loss, SupConLoss

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser(description='PyTorch implementation of Partial label continual learning')

parser.add_argument('--method', type=str, default='mir_replay', choices=['rand_replay', 'mir_replay'])
parser.add_argument('--dataset', type=str, default='split_cifar10',
                    choices=['split_mnist', 'permuted_mnist', 'split_cifar10'])
parser.add_argument('--buffer_batch_size', type=int, default=10)
parser.add_argument('--mem_size', type=int, default=50)
parser.add_argument('--subsample', type=int, default=50,
                    help="for subsampling in --method=replay, set to 0 to disable")
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--exp-dir', default='experiment/PiCO_MIR', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture')
# parser.add_argument('-j', '--workers', default=32, type=int,
#                     help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=75, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     help='mini-batch size (default: 256), this is the total '
#                          'batch size of all GPUs on the current node when '
#                          'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=74, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=123, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int,
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=5, type=int,
                    help='Start Prototype Updating')
parser.add_argument('--partial_rate', default=0.3, type=float,
                    help='ambiguity level (q)')
parser.add_argument('--hierarchical', action='store_true',
                    help='for CIFAR-100 fine-grained training')
parser.add_argument('--compare_to_old_logits', action='store_true', help='uses old logits')

# distributed training
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://localhost:10002', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')

args = parser.parse_args(args=[])
args.device = 'cuda'
args.cuda = torch.cuda.is_available()


class Buffer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.place_left = True
        input_size = args.input_size

        buffer_size = args.mem_size  # 50 * 10 classes = 500
        print(f'buffer has {buffer_size} slots.')

        buffer_sample_weak = torch.FloatTensor(buffer_size, *input_size).fill_(0).to(args.device)
        buffer_sample_strong = torch.FloatTensor(buffer_size, *input_size).fill_(0).to(args.device)
        buffer_label = torch.LongTensor(buffer_size).fill_(0).to(args.device)
        buffer_partial = torch.LongTensor(buffer_size, args.n_classes).fill_(0).to(args.device)
        buffer_task = torch.LongTensor(buffer_size).fill_(0).to(args.device)
        buffer_indices = torch.LongTensor(buffer_size).fill_(0).to(args.device)
        logits = torch.FloatTensor(buffer_size, args.n_classes).fill_(0).to(args.device)

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full = 0

        self.register_buffer('buffer_sample_weak', buffer_sample_weak)
        self.register_buffer('buffer_sample_strong', buffer_sample_strong)
        self.register_buffer('buffer_label', buffer_label)
        self.register_buffer('buffer_partial', buffer_partial)
        self.register_buffer('buffer_task', buffer_task)
        self.register_buffer('buffer_indices', buffer_indices)
        self.register_buffer('logits', logits)

        self.to_one_hot = lambda x: x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)

    @property
    def sample(self):
        return self.buffer_sample_weak[:self.current_index], self.buffer_sample_strong[:self.current_index]

    def sample(self, batch_size, exclude_task=None, return_idx=False):
        if exclude_task is not None:
            valid_indices = (self.test != exclude_task)
            valid_indices = valid_indices.nonzero().squeeze()
            bxw, bxs, by, bp, bt, bi = self.buffer_sample_weak[valid_indices], self.buffer_sample_strong[valid_indices], \
                                       self.buffer_label[valid_indices], self.buffer_partial[valid_indices], \
                                       self.buffer_task[valid_indices], self.buffer_indices[valid_indices]
        else:
            bxw, bxs, by, bp, bt, bi = self.buffer_sample_weak[: self.current_index], self.buffer_sample_strong[
                                                                                      : self.current_index], self.buffer_label[
                                                                                                             : self.current_index], self.buffer_partial[
                                                                                                                                    : self.current_index], self.buffer_task[
                                                                                                                                                           : self.current_index], self.buffer_indices[
                                                                                                                                                                                  : self.current_index]

        indices = torch.from_numpy(np.random.choice(bxw.size(0), batch_size, replace=False)).to(self.args.device)
        if return_idx:
            return bxw[indices], bxs[indices], by[indices], bp[indices], bt[indices], bi[indices], indices
        else:
            return bxw[indices], bxs[indices], by[indices], bp[indices], bt[indices], bi[indices]

    @property
    def label(self):
        return self.to_one_hot(self.buffer_label[: self.current_index])

    @property
    def test(self):
        return self.buffer_task[: self.current_index]

    def add_reservoir(self, sample_weak, sample_strong, label, partial, task, index):
        n_elem = sample_weak.size(0)
        place_left = max(0, self.buffer_sample_weak.size(0) - self.current_index)

        if place_left:
            offset = min(place_left, n_elem)
            self.buffer_sample_weak[self.current_index: self.current_index + offset].data.copy_(sample_weak[:offset])
            self.buffer_sample_strong[self.current_index: self.current_index + offset].data.copy_(
                sample_strong[:offset])
            self.buffer_label[self.current_index: self.current_index + offset].data.copy_(label[:offset])
            self.buffer_partial[self.current_index: self.current_index + offset].data.copy_(partial[:offset])
            self.buffer_task[self.current_index: self.current_index + offset].fill_(task)
            self.buffer_indices[self.current_index: self.current_index + offset].data.copy_(index[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            if offset == n_elem:
                return

        self.place_left = False
        sample_weak, sample_strong, label, partial = sample_weak[place_left:], sample_strong[place_left:], label[
                                                                                                           place_left:], partial[
                                                                                                                         place_left:]

        indices = torch.FloatTensor(sample_weak.size(0)).to(sample_weak.device).uniform_(0, self.n_seen_so_far).long()
        vlid_indeces = (indices < self.buffer_sample_weak.size(0)).long()

        idx_new_data = vlid_indeces.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        self.n_seen_so_far += sample_weak.size(0)

        if idx_buffer.numel() == 0:
            return

        self.buffer_sample_weak[idx_buffer] = sample_weak[idx_new_data]
        self.buffer_sample_strong[idx_buffer] = sample_strong[idx_new_data]
        self.buffer_label[idx_buffer] = label[idx_new_data]
        self.buffer_partial[idx_buffer] = partial[idx_new_data]
        self.buffer_task[idx_buffer] = task
        self.buffer_indices[idx_buffer] = index[idx_new_data]


class CIFAR10_Augmentation(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix

        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]
        )
        self.strong_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]
        )

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, idx):
        each_image_weak = self.weak_transform(self.images[idx])
        each_image_strong = self.strong_transform(self.images[idx])
        each_label = self.given_label_matrix[idx]
        each_true_label = self.true_labels[idx]

        return each_image_weak, each_image_strong, each_label, each_true_label, idx


class MNIST_Augmentation(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix

        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
        self.strong_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                # RandomAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, idx):
        each_image_weak = self.weak_transform(self.images[idx])
        each_image_strong = self.strong_transform(self.images[idx])
        each_label = self.given_label_matrix[idx]
        each_true_label = self.true_labels[idx]

        return each_image_weak, each_image_strong, each_label, each_true_label, idx


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, **kwargs):
        self.x, self.y = x, y

        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        if type(x) != torch.Tensor:
            x = self.transform(Image.open(x).convert('RGB'))
            y = torch.Tensor(1).fill_(y).long().squeeze()
        else:
            x = x.float() / 255.
            y = y.long()

        if self.source == 'mnist':
            return x, y
        else:
            return (x - 0.5) * 2, y  # normalization


class XYZDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, z, **kwargs):
        self.x, self.y, self.z = x, y, z
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y, z = self.x[idx], self.y[idx], self.z[idx]

        if type(x) != torch.Tensor:
            x = self.transform(Image.open(x).convert('RGB'))
            y = torch.Tensor(1).fill_(y).long().squeeze()
            z = torch.Tensor(10).fill_(z).long().squeeze()
        else:
            x = x.float() / 255.
            y = y.long()
            z = z.long()

        if self.source == 'mnist':
            return x, y, z
        else:
            return (x - 0.5) * 2, y, z  # normalization


class CLDataLoader(object):
    def __init__(self, datasets_per_task, args, train=True):
        batch_size = 10 if train else 24

        self.datasets = datasets_per_task
        if train:
            self.loaders = [
                torch.utils.data.DataLoader(dataset=x,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True,
                                            drop_last=True,
                                            prefetch_factor=24)
                for x in self.datasets
            ]
        else:
            self.loaders = [
                torch.utils.data.DataLoader(dataset=x, batch_size=batch_size * 2, shuffle=False,
                                            num_workers=4)
                for x in self.datasets
            ]

    def __getitem__(self, item):
        return self.loaders[item]

    def __len__(self):
        return len(self.loaders)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


def get_logger(names, n_runs=1, n_tasks=None):
    log = OD()
    for i in range(n_runs):
        log[i] = {}
        for mode in ['train', 'valid', 'test']:
            log[i][mode] = {}
            for name in names:
                log[i][mode][name] = np.zeros([n_tasks, n_tasks])
            log[i][mode]['final_acc'] = 0.
            log[i][mode]['final_forget'] = 0.
    return log


def make_valid_from_train(dataset):
    train_ds, valid_ds, partial_set = [], [], []
    for task in dataset:
        samples, labels, partial = task
        shuffle_perm = torch.randperm(len(samples))
        samples = samples[shuffle_perm]
        labels = labels[shuffle_perm]
        partial = partial[shuffle_perm]
        split = int(len(samples) * 0.95)
        train_samples, train_labels, train_partial = samples[: split], labels[: split], partial[: split]
        valid_samples, valid_labels = samples[split:], labels[split:]
        train_ds.append((train_samples, train_labels, train_partial))
        valid_ds.append((valid_samples, valid_labels))
        partial_set.append(train_partial)
    return train_ds, valid_ds, partial_set


def get_split_cifar10(partial_rate, args):
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ]
    )

    args.n_tasks = 5
    args.n_classes = 10
    args.buffer_size = args.n_tasks * args.mem_size * 2
    args.n_classes_per_task = 2
    args.input_size = [3, 32, 32]  # channels, height, width

    # why we need to do this?
    # args.output_loss = 'mse'
    # print('\nsetting output loss to MSE')

    temp_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    # data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform)

    try:
        train_samples, train_labels = temp_train.data, torch.Tensor(temp_train.targets).long()
        test_samples, test_labels = test_dataset.data, test_dataset.targets
    except:
        train_samples, train_labels = temp_train.train_data, torch.Tensor(temp_train.train_labels).long()
        test_samples, test_labels = test_dataset.test_data, test_dataset.test_labels

    partialY = generate_uniform_cv_candidate_labels(train_labels, partial_rate)

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), train_labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    print('Average candidate num: ', partialY.sum(1).mean())

    # partial_matrix_dataset = CIFAR10_Augmentation(train_samples, partialY.float(), train_labels.float())

    # sort the dataset
    train_data = [
        (x, y, z) for (x, y, z) in sorted(zip(train_samples, train_labels, partialY), key=lambda x: x[1])
    ]
    test_data = [
        (x, y) for (x, y) in sorted(zip(test_samples, test_labels), key=lambda x: x[1])
    ]

    train_samples, train_labels, partialY = [
        np.stack([elem[i] for elem in train_data]) for i in [0, 1, 2]
    ]

    test_samples, test_labels = [
        np.stack([elem[i] for elem in test_data]) for i in [0, 1]
    ]

    train_samples = torch.Tensor(train_samples).permute(0, 3, 1, 2).contiguous()
    test_samples = torch.Tensor(test_samples).permute(0, 3, 1, 2).contiguous()
    partialY = torch.Tensor(partialY).long()
    train_labels = torch.Tensor(train_labels).long()
    test_labels = torch.Tensor(test_labels).long()

    train_idx = [((train_labels + i) % 10).argmax() for i in range(10)]
    test_idx = [((test_labels + i) % 10).argmax() for i in range(10)]
    train_idx.sort()
    test_idx.sort()

    train_ds, test_ds = [], []
    skip = 2
    for i in range(0, 10, skip):
        if i == 8:
            train_s = train_idx[i]
            test_s = test_idx[i]
            train_ds.append((train_samples[train_s:], train_labels[train_s:], partialY[train_s:]))
            test_ds.append((test_samples[test_s:], test_labels[test_s:]))
        else:
            train_s, train_e = train_idx[i], train_idx[i + skip]
            test_s, test_e = test_idx[i], test_idx[i + skip]

            train_ds.append(
                (train_samples[train_s: train_e], train_labels[train_s: train_e], partialY[train_s: train_e]))
            test_ds.append((test_samples[test_s: test_e], test_labels[test_s: test_e]))

    train_ds, valid_ds, partial_set = make_valid_from_train(train_ds)
    train_ds = map(lambda x: CIFAR10_Augmentation(x[0], x[2], x[1]), train_ds)
    valid_ds = map(lambda x: XYDataset(x[0], x[1], **{'source': 'cifar10'}), valid_ds)
    test_ds = map(lambda x: XYDataset(x[0], x[1], **{'source': 'cifar10'}), test_ds)

    return [train_ds, valid_ds, test_ds], partial_set


def get_split_mnist(partial_rate, args):
    args.n_classes = 10
    args.n_tasks = 5
    if 'mem_size' in args:
        args.buffer_size = args.n_tasks * args.mem_size * 2
    args.use_conv = False
    args.input_type = 'binary'
    args.input_size = [1, 28, 28]
    # if args.output_loss is None:
    #     args.output_loss = 'bernouilli'

    assert args.n_tasks in [5, 10], 'SplitMnist only works with 5 or 10 tasks'
    assert '1.' in str(torch.__version__)[:2], 'Use Pytorch 1.x!'

    # fetch MNIST
    train = datasets.MNIST('./data', train=True, download=True)
    test = datasets.MNIST('./data', train=False, download=True)

    try:
        train_x, train_y = train.data, train.targets
        test_x, test_y = test.data, test.targets
    except:
        train_x, train_y = train.train_data, train.train_labels
        test_x, test_y = test.test_data, test.test_labels

    partialY = generate_uniform_cv_candidate_labels(train_y, partial_rate)

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), train_y] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    print('Average candidate num: ', partialY.sum(1).mean())

    # sort according to the label
    out_train = [
        (x, y, z) for (x, y, z) in sorted(zip(train_x, train_y, partialY), key=lambda x: x[1])
    ]

    out_test = [
        (x, y) for (x, y) in sorted(zip(test_x, test_y), key=lambda x: x[1])
    ]

    train_x, train_y, partialY = [
        np.stack([elem[i] for elem in out_train]) for i in [0, 1, 2]
    ]


    test_x, test_y = [
        np.stack([elem[i] for elem in out_test]) for i in [0, 1]
    ]

    # if args.use_conv:
    #    train_x = train_x.unsqueeze(1)
    #    test_x  = test_x.unsqueeze(1)
    # else:
    #    train_x = train_x.view(train_x.size(0), -1)
    #    test_x  = test_x.view(test_x.size(0), -1)

    # cast in 3D:
    train_x = torch.tensor(train_x)
    test_x = torch.tensor(test_x)
    train_x = train_x.view(train_x.size(0), 1, train_x.size(1), train_x.size(2))
    test_x = test_x.view(test_x.size(0), 1, test_x.size(1), test_x.size(2))
    partialY = torch.Tensor(partialY).long()
    train_y = torch.Tensor(train_y).long()
    test_y = torch.Tensor(test_y).long()

    # get indices of class split
    train_idx = [((train_y + i) % 10).argmax() for i in range(10)]
    train_idx = [0] + sorted(train_idx)

    test_idx = [((test_y + i) % 10).argmax() for i in range(10)]
    test_idx = [0] + sorted(test_idx)

    train_ds, test_ds = [], []
    skip = 10 // args.n_tasks
    for i in range(0, 10, skip):
        tr_s, tr_e = train_idx[i], train_idx[i + skip]
        te_s, te_e = test_idx[i], test_idx[i + skip]

        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e], partialY[tr_s:tr_e])]
        test_ds += [(test_x[te_s:te_e], test_y[te_s:te_e])]

    train_ds, val_ds, partial_set = make_valid_from_train(train_ds)

    train_ds = map(lambda x: MNIST_Augmentation(x[0], x[2], x[1]), train_ds)
    val_ds = map(lambda x: XYDataset(x[0], x[1], **{'source': 'mnist'}), val_ds)
    test_ds = map(lambda x: XYDataset(x[0], x[1], **{'source': 'mnist'}), test_ds)

    return [train_ds, val_ds, test_ds], partial_set


def get_grad_vector(args, pp, grad_dims):
    grads = torch.Tensor(sum(grad_dims))
    if args.cuda: grads = grads.cuda()

    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


def overwrite_grad(pp, new_grad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        param.grad = torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(
            param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1


def get_future_step_parameters(this_net, grad_vector, grad_dims, lr=1):
    """
    computes \theta-\delta\theta
    :param this_net:
    :param grad_vector:
    :return:
    """
    torch.save(this_net, 'temp.pkl')
    new_net = PiCO(args, SupConResNet)
    new_net = torch.load('temp.pkl')
    overwrite_grad(new_net.parameters, grad_vector, grad_dims)
    with torch.no_grad():
        for param in new_net.parameters():
            if param.grad is not None:
                param.data = param.data - lr * param.grad.data
    return new_net


def train(train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, args, tb_logger, task, buffer,
          start_upd_prot=False, rehearse=True):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    buffer_loss_log = AverageMeter('Loss@Buffer', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_proto, loss_cls_log, loss_cont_log, buffer_loss_log],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()

    end = time.time()
    for i, (images_w, images_s, labels, true_labels, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
        Y_true = true_labels.long().detach().cuda()
        # for showing training accuracy and will not be used when training

        cls_out, features_cont, pseudo_score_cont, partial_target_cont, score_prot \
            = model(X_w, X_s, Y, args)
        batch_size = cls_out.shape[0]
        pseudo_target_max, pseudo_target_cont = torch.max(pseudo_score_cont, dim=1)
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y,
                                      task=torch.Tensor(index.shape[0]).fill_(task).long().cuda())
            # warm up ended

        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
            # get positive set by contrasting predicted labels
        else:
            mask = None
            # Warmup using MoCo

        # contrastive loss
        loss_cont = loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size)
        # classification loss
        loss_cls = loss_fn(cls_out, index, torch.Tensor(index.shape[0]).fill_(task).long().cuda())

        loss = loss_cls + args.loss_weight * loss_cont
        loss_cls_log.update(loss_cls.item())
        loss_cont_log.update(loss_cont.item())

        # log accuracy
        acc = accuracy(cls_out, Y_true)[0]
        acc_cls.update(acc[0])
        acc = accuracy(score_prot, Y_true)[0]
        acc_proto.update(acc[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if not rehearse:
            optimizer.step()
        else:
            if args.method == 'mir_replay':
                buffer_x_weak, buffer_x_strong, buffer_y, buffer_p, buffer_t, buffer_i, subsample = buffer.sample(
                    args.subsample, exclude_task=task, return_idx=True)
                grad_dims = []
                for param in model.parameters():
                    grad_dims.append(param.data.numel())
                grad_vector = get_grad_vector(args, model.parameters, grad_dims)
                model_temp = get_future_step_parameters(model, grad_vector, grad_dims, lr=args.lr)

                with torch.no_grad():
                    mem_cls_out_pre, mem_features_cont_pre, mem_pseudo_score_cont_pre, mem_partial_target_cont_pre, mem_score_prot_pre = model(
                        buffer_x_weak, buffer_x_strong, buffer_p, args, update_queue=False)
                    mem_cls_out_post, mem_features_cont_post, mem_pseudo_score_cont_post, mem_partial_target_cont_post, mem_score_prot_post = model_temp(
                        buffer_x_weak, buffer_x_strong, buffer_p, args, update_queue=False)

                    pseudo_target_max_pre, pseudo_target_cont_pre = torch.max(mem_pseudo_score_cont_pre, dim=1)
                    pseudo_target_cont_pre = pseudo_target_cont_pre.contiguous().view(-1, 1)

                    pseudo_target_max_post, pseudo_target_cont_post = torch.max(mem_pseudo_score_cont_post, dim=1)
                    pseudo_target_cont_post = pseudo_target_cont_post.contiguous().view(-1, 1)

                    if start_upd_prot:
                        mask_pre = torch.eq(pseudo_target_cont_pre[:args.subsample],
                                            pseudo_target_cont_pre.T).float().cuda()
                        mask_post = torch.eq(pseudo_target_cont_post[:args.subsample],
                                             pseudo_target_cont_post.T).float().cuda()
                        # get positive set by contrasting predicted labels
                    else:
                        mask_pre = None
                        mask_post = None
                        # Warmup using MoCo

                    mem_loss_cont_pre = loss_cont_fn(features=mem_features_cont_pre, mask=mask_pre,
                                                     batch_size=args.subsample, memory=True)
                    mem_loss_cont_post = loss_cont_fn(features=mem_features_cont_post, mask=mask_post,
                                                      batch_size=args.subsample, memory=True)

                    mem_loss_cls_pre = loss_fn(mem_cls_out_pre, buffer_i, buffer_t, memory=True)
                    mem_loss_cls_post = loss_fn(mem_cls_out_post, buffer_i, buffer_t, memory=True)

                    loss_pre = mem_loss_cls_pre + args.loss_weight * mem_loss_cont_pre
                    loss_post = mem_loss_cls_post + args.loss_weight * mem_loss_cont_post
                    scores = loss_post - loss_pre

                    all_logits = scores
                    big_ind = all_logits.sort(descending=True)[1][:args.buffer_batch_size]

                    # idx = subsample[big_ind]

                buffer_x_weak, buffer_x_strong, buffer_y, buffer_p, buffer_t, buffer_i = buffer_x_weak[big_ind], \
                                                                                         buffer_x_strong[big_ind], \
                                                                                         buffer_y[big_ind], buffer_p[
                                                                                             big_ind], buffer_t[
                                                                                             big_ind], buffer_i[big_ind]
            else:
                buffer_x_weak, buffer_x_strong, buffer_y, buffer_p, buffer_t, buffer_i = buffer.sample(
                    args.buffer_batch_size, exclude_task=task)

            buffer_cls_out, buffer_features_cont, buffer_pseudo_score_cont, buffer_partial_target_cont, buffer_score_prot = model(
                buffer_x_weak, buffer_x_strong, buffer_p, args)
            buffer_batch_size = buffer_cls_out.shape[0]
            buffer_pseudo_target_max, buffer_pseudo_target_cont = torch.max(buffer_pseudo_score_cont, dim=1)
            buffer_pseudo_target_cont = buffer_pseudo_target_cont.contiguous().view(-1, 1)

            if start_upd_prot:
                loss_fn.confidence_update(temp_un_conf=buffer_score_prot, batch_index=buffer_i, batchY=buffer_p,
                                          task=buffer_t)

            if start_upd_prot:
                buffer_mask = torch.eq(buffer_pseudo_target_cont[:buffer_batch_size],
                                       buffer_pseudo_target_cont.T).float().cuda()
            else:
                buffer_mask = None

            buffer_loss_cont = loss_cont_fn(features=buffer_features_cont, mask=buffer_mask,
                                            batch_size=buffer_batch_size)
            buffer_loss_cls = loss_fn(buffer_cls_out, buffer_i, buffer_t)
            loss = buffer_loss_cls + args.loss_weight * buffer_loss_cont
            # loss_cls_log.update(buffer_loss_cls.item())
            # loss_cont_log.update(buffer_loss_cont.item())
            buffer_loss_log.update(loss.item())
            buffer_y = buffer_y.long().detach().cuda()
            buffer_acc = accuracy(buffer_cls_out, buffer_y)[0]
            # acc_cls.update(buffer_acc[0])
            buffer_acc = accuracy(buffer_score_prot, buffer_y)[0]
            # acc_proto.update(buffer_acc[0])
            loss.backward()
            optimizer.step()

        if epoch == 0:
            buffer.add_reservoir(X_w, X_s, Y_true, Y, task, index)

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.display(i)
        elif i == len(train_loader) - 1:
            progress.display(i)

        end = time.time()

    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss', loss_cont_log.avg, epoch)


def test(model, test_loader, args, epoch, tb_logger, task):
    with torch.no_grad():
        print(f'==> Evaluating Task{task}...')
        model.eval()
        top1_acc = AverageMeter('Top1')
        top5_acc = AverageMeter('Top5')
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images, args, eval_only=True)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(args.gpu)
        # dist.all_reduce(acc_tensors)
        # acc_tensors /= args.world_size

        print('Accuracy is %.2f%% (%.2f%%)' % (acc_tensors[0], acc_tensors[1]))
        if args.gpu == 0:
            tb_logger.log_value('Top1 Acc', acc_tensors[0], epoch)
            tb_logger.log_value('Top5 Acc', acc_tensors[1], epoch)
    return acc_tensors[0]


data, partial_set = get_split_cifar10(args.partial_rate, args)
train_loader, val_loader, test_loader = [CLDataLoader(elem, args, train=t) for elem, t in
                                         zip(data, [True, False, False])]
args.mem_size = args.mem_size * args.n_classes
LOG = get_logger(['total_loss', 'cont_loss', 'cls_loss', 'acc'], n_runs=args.n_runs, n_tasks=args.n_tasks)

args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
iterations = args.lr_decay_epochs.split(',')
args.lr_decay_epochs = list([])
for it in iterations:
    args.lr_decay_epochs.append(int(it))
print(args)

model_path = 'ds_{ds}_pr_{pr}_lr_{lr}_ep_{ep}_ps_{ps}_lw_{lw}_pm_{pm}_arch_{arch}_heir_{heir}_sd_{seed}'.format(
    ds=args.dataset,
    pr=args.partial_rate,
    lr=args.lr,
    ep=args.epochs,
    ps=args.prot_start,
    lw=args.loss_weight,
    pm=args.proto_m,
    arch=args.arch,
    seed=args.seed,
    heir=args.hierarchical)

# args.exp_dir = os.path.join(args.exp_dir, model_path)
# if not os.path.exists(args.exp_dir):
#     os.makedirs(args.exp_dir)

for run in range(args.n_runs):
    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    if args.device is not None:
        print("Use GPU: {} for training".format(args.device))

    print("=> creating model '{}'".format(args.arch))

    model = PiCO(args, SupConResNet)
    torch.cuda.set_device(0)
    model.to(args.device)
    # model.fc = torch.nn.Linear(torchvision.models.resnet18().fc.in_features, args.n_classes)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    buffer = Buffer(args)

    if run == 0:
        print(f'number of classifier parameters: {sum([np.prod(p.size()) for p in model.parameters()])}')
        print(f'buffer parameters: {np.prod(buffer.buffer_sample_weak.size())}')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader, train_givenY, test_loader = train_loader, partial_set, test_loader

    print('Calculating uniform targets...')
    confidence = []
    for i in range(len(train_givenY)):
        tempY = train_givenY[i].sum(dim=1).unsqueeze(1).repeat(1, train_givenY[i].shape[1])
        temp_conf = train_givenY[i] / tempY
        temp_conf = temp_conf.to(args.device)
        confidence.append(temp_conf)
    # padding all confidence in the same size
    max_len = max([len(conf) for conf in confidence])
    for i in range(len(confidence)):
        confidence[i] = torch.cat([confidence[i], torch.zeros(max_len - len(confidence[i]), args.n_classes).to(args.device)],
                                  dim=0)
    confidence = torch.stack(confidence, dim=0)

    loss_fn = partial_loss(confidence)
    loss_cont_fn = SupConLoss()

    print('\nStart training...\n')

    for task, tr_loader in enumerate(train_loader):
        dir = os.path.join(args.exp_dir, f'task_{task}', model_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if args.gpu == 0:
            logger = tb_logger.Logger(logdir=os.path.join(dir, 'tensorboard'), flush_secs=2)
        else:
            logger = None
        # best_acc = 0
        # mmc = 0
        rehearse = task > 0
        for epoch in range(args.start_epoch, args.epochs):
            if epoch == 0:
                print('\n--------------------------------------')
                print('Run #{} Task #{}--> Train Classifier'.format(
                    run, task))
                print('--------------------------------------\n')

            print(f'Epoch: {epoch}')
            is_best = False
            start_upd_prot = epoch >= args.prot_start
            rehearse = task > 0

            adjust_learning_rate(args, optimizer, epoch)
            train(tr_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, args, logger, task, buffer, start_upd_prot,
                  rehearse=rehearse)
            loss_fn.set_conf_ema_m(epoch, args)


            save_checkpoint({
                'task': task,
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(dir),
                best_file_name='{}/checkpoint_best.pth.tar'.format(dir))

        for te_task, te_loader in enumerate(test_loader):
            if te_task > task: break
            acc_test = test(model, te_loader, args, epoch, logger, te_task)

            # mmc = loss_fn.confidence.max(dim=1)[0].mean()

        # ------------------------ eval ------------------------ #
        # eval_loader = [('valid', val_loader), ('test', test_loader)]
        #
        # for te_task, te_loader in enumerate(val_loader):
        #     if te_task > task: break
        #     print(f'Validation Run #{run} Task #{te_task} --> Validation Classifier')
        #
        #     acc_test = test(model, te_loader, args, epoch, logger, te_task)

    # print('--------------------------------------')
    # print(f'Run #{run} Final Results')
    # print('--------------------------------------')
    # for mode in ['valid', 'test']:
    #     final_accuracy = LOG[run][mode]['acc'][:, task]
    #     logging_per_task(LOG, run, mode, 'final_acc', task, value=np.round(np.mean(final_accuracy), 2))
    #
    #     best_acc = np.max(LOG[run][mode]['acc'], 1)
    #     final_forgets = best_acc - LOG[run][mode]['acc'][:, task]
    #     logging_per_task(LOG, run, mode, 'final_forget', task, value=np.round(np.mean(final_forgets[:-1]), 2))
    #
    #     print(f'\n{mode}:')
    #     print('final accuracy: {}'.format(final_accuracy))
    #     print('average: {}'.format(LOG[run][mode]['final_acc']))
    #     print('final forgetting: {}'.format(final_forgets))
    #     print('average: {}\n'.format(LOG[run][mode]['final_forget']))

os.system('shutdown')