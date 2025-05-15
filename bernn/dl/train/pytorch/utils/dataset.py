#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import csv
from PIL import Image
import torch
import itertools
import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd
from sklearn.preprocessing import minmax_scale as scale
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, StandardScaler
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torch import nn

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class MSDataset3(Dataset):
    def __init__(self, data, meta, names=None, labels=None, batches=None, sets=None, transform=None, crop_size=-1,
                 add_noise=False, random_recs=False, triplet_dloss=False):
        """

        Args:
            data: Contains a dict of data
            meta: array of meta data
            names: array or list of names
            labels: array or list of labels
            batches: array or list of batches
            sets: array or list of sets
            transform: transform to apply to the data
            crop_size: crop size to apply to the data
            add_noise: Whether to add noise to the data
            random_recs: Whether to sample random reconstructions
            triplet_dloss: Whether to use triplet loss of the domain
        """
        self.random_recs = random_recs
        try:
            self.samples = data.to_numpy()
        except:
            self.samples = data

        self.add_noise = add_noise
        self.names = names
        self.meta = meta
        self.sets = sets
        self.transform = transform
        self.crop_size = crop_size
        self.labels = labels
        self.unique_labels = list(set(labels))
        self.batches = batches
        self.unique_batches = np.unique(batches)
        labels_inds = {label: [i for i, x in enumerate(labels) if x == label] for label in self.unique_labels}
        batches_inds = {batch: [i for i, x in enumerate(batches) if x == batch] for batch in self.unique_batches}
        # try:
        try:
            self.labels_data = {label: data.iloc[labels_inds[label]].to_numpy() for label in labels}
            self.labels_meta_data = {label: meta.iloc[labels_inds[label]].to_numpy() for label in labels}
            self.batches_data = {batch: data.iloc[batches_inds[batch]].to_numpy() for batch in batches}
            self.batches_meta_data = {batch: meta.iloc[batches_inds[batch]].to_numpy() for batch in batches}
        except:
            self.labels_data = {label: data[labels_inds[label]] for label in labels}
            self.labels_meta_data = {label: meta[labels_inds[label]] for label in labels}
            self.batches_data = {batch: data[batches_inds[batch]] for batch in batches}
            self.batches_meta_data = {batch: meta[batches_inds[batch]] for batch in batches}

        # except:
        #     print(labels)
        self.n_labels = {label: len(self.labels_data[label]) for label in labels}
        self.n_batches = {batch: len(self.batches_data[batch]) for batch in batches}
        self.triplet_dloss = triplet_dloss

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta_to_rec = None
        if self.labels is not None:
            label = self.labels[idx]
            batch = self.batches[idx]
            set = self.sets[idx]
            try:
                name = self.names[idx]
            except:
                name = str(self.names.iloc[idx])
            try:
                meta_to_rec = self.meta[idx]
            except:
                meta_to_rec = self.meta.iloc[idx].to_numpy()

        else:
            label = None
            batch = None
            name = None
        if self.random_recs:
            to_rec = self.labels_data[label][np.random.randint(0, self.n_labels[label])].copy()
            not_label = None
            while not_label == label or not_label is None:
                not_label = self.unique_labels[np.random.randint(0, len(self.unique_labels))].copy()
            ind = np.random.randint(0, self.n_labels[not_label])
            not_to_rec = self.labels_data[not_label][ind].copy()
            meta_not_to_rec = self.labels_meta_data[not_label][ind].copy()
            meta_to_rec = self.meta[idx]
        else:
            to_rec = self.samples[idx].copy()
            not_to_rec = torch.Tensor([0])
        if (self.triplet_dloss == 'revTriplet' or self.triplet_dloss == 'inverseTriplet') and len(self.unique_batches) > 1:
            not_batch_label = None
            while not_batch_label == batch or not_batch_label is None:
                not_batch_label = self.unique_batches[np.random.randint(0, len(self.unique_batches))].copy()
            pos_ind = np.random.randint(0, self.n_batches[batch])
            neg_ind = np.random.randint(0, self.n_batches[not_batch_label])
            pos_batch_sample = self.batches_data[batch][pos_ind].copy()
            neg_batch_sample = self.batches_data[not_batch_label][neg_ind].copy()
            meta_pos_batch_sample = self.batches_meta_data[batch][pos_ind].copy()
            meta_neg_batch_sample = self.batches_meta_data[not_batch_label][neg_ind].copy()
        else:
            pos_batch_sample = self.samples[idx].copy()
            neg_batch_sample = self.samples[idx].copy()
            meta_pos_batch_sample = self.samples[idx].copy()
            meta_neg_batch_sample = self.samples[idx].copy()
        x = self.samples[idx]
        if self.crop_size != -1:
            max_start_crop = x.shape[1] - self.crop_size
            ran = np.random.randint(0, max_start_crop)
            x = torch.Tensor(x)[:, ran:ran + self.crop_size]  # .to(device)
        if self.transform:
            x = self.transform(np.expand_dims(x, 0)).squeeze()
            to_rec = self.transform(np.expand_dims(to_rec, 0)).squeeze()
            not_to_rec = self.transform(np.expand_dims(not_to_rec, 0)).squeeze()
            pos_batch_sample = self.transform(np.expand_dims(pos_batch_sample, 0)).squeeze()
            neg_batch_sample = self.transform(np.expand_dims(neg_batch_sample, 0)).squeeze()

        if self.add_noise:
            if np.random.random() > 0.5:
                x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        return x, meta_to_rec, name, label, batch, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample, \
            meta_pos_batch_sample, meta_neg_batch_sample, set


# This function is much faster than using pd.read_csv
def load_data(path):
    cols = csv.DictReader(open(path))
    data = []
    names = []
    for i, row in enumerate(cols):
        names += [list(row.values())[0]]
        data += [np.array(list(row.values())[1:], dtype=float)]
    data = np.stack(data)
    # data = pd.DataFrame(data, index=labels, columns=list(row.keys())[1:])
    labels = np.array([d.split('_')[1].split('-')[0] for d in names])
    batches = np.array([d.split('_')[0] for d in names])
    # data = get_normalized(torch.Tensor(np.array(data.values, dtype='float')))
    data[np.isnan(data)] = 0
    columns = list(row.keys())[1:]
    return data.T, names, labels, batches, columns


def load_checkpoint(checkpoint_path,
                    model,
                    optimizer,
                    name,
                    ):
    losses = {
        "train": [],
        "valid": [],
    }
    if name not in os.listdir(checkpoint_path):
        print("checkpoint not found...")
        return model, None, 1, losses, None, None, {'valid_loss': -1}
    checkpoint_dict = torch.load(checkpoint_path + '/' + name, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    best_values = checkpoint_dict['best_values']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    state_dict = checkpoint_dict['model']
    model.load_state_dict(state_dict)
    try:
        losses_recon = checkpoint_dict['losses_recon']
        kl_divs = checkpoint_dict['kl_divs']
    except:
        losses_recon = None
        kl_divs = None

    losses = checkpoint_dict['losses']
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return model, optimizer, epoch, losses, kl_divs, losses_recon, best_values, checkpoint_dict['finished']


def final_save_checkpoint(checkpoint_path, model, optimizer, name):
    model, optimizer, epoch, losses, _, _, best_values, _ = load_checkpoint(checkpoint_path, model, optimizer, name)
    torch.save({'model': model.state_dict(),
                'losses': losses,
                'best_values': best_values,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                # 'learning_rate': learning_rate,
                'finished': True},
               checkpoint_path + '/' + name)


def save_checkpoint(model,
                    optimizer,
                    # learning_rate,
                    epoch,
                    checkpoint_path,
                    losses,
                    best_values,
                    name="cnn",
                    ):
    # model_for_saving = model_name(input_shape=input_shape, nb_classes=nb_classes, variant=variant,
    #                               activation=activation)
    # model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model.state_dict(),
                'losses': losses,
                'best_values': best_values,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                # 'learning_rate': learning_rate,
                'finished': False},
               checkpoint_path + '/' + name)


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]


class validation_spliter:
    def __init__(self, dataset, cv):
        self.cv = cv
        self.dataset = dataset
        self.current_cv = 0
        self.val_offset = int(np.floor(len(self.dataset) / self.cv))
        self.current_pos = 0

    def __next__(self):
        self.current_cv += 1
        # if self.current_cv == self.cv:
        #     val_offset = len(self.dataset) - self.current_pos
        # else:
        #     val_offset = self.val_offset
        partial_dataset = PartialDataset(self.dataset, 0, self.val_offset), \
            PartialDataset(self.dataset, self.val_offset, len(self.dataset) - self.val_offset)

        # Move the samples currently used for the validation set at the end for the next split
        tmp = self.dataset.samples[:self.val_offset]
        self.dataset.samples = np.concatenate([self.dataset.samples[self.val_offset:], tmp], 0)

        return partial_dataset


def get_loaders(data, random_recs, samples_weights, triplet_dloss, ae=None, classifier=None, bs=64, device='cuda'):
    """

    Args:
        data:
        ae:
        classifier:

    Returns:

    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(0.5, 0.5),
        # torchvision.transforms.Normalize(np.mean(data['inputs']['train'].to_numpy().reshape(1, -1)), np.std(data['inputs']['train'].to_numpy().reshape(1, -1))),
    ])

    train_set = MSDataset3(data['inputs']['train'], data['meta']['train'], data['names']['train'].to_numpy(),
                           data['cats']['train'], [x for x in data['batches']['train']],
                           [x for x in data['sets']['train']], transform=transform, crop_size=-1,
                           random_recs=random_recs, triplet_dloss=triplet_dloss)
    train_pool_set = MSDataset3(data['inputs']['train_pool'], data['meta']['train_pool'], data['names']['train_pool'],
                                data['cats']['train_pool'], [x for x in data['batches']['train_pool']],
                                [x for x in data['sets']['train_pool']], transform=transform, crop_size=-1,
                                random_recs=False, triplet_dloss=triplet_dloss)
    valid_pool_set = MSDataset3(data['inputs']['valid_pool'], data['meta']['valid_pool'], data['names']['valid_pool'],
                                data['cats']['valid_pool'], [x for x in data['batches']['valid_pool']],
                                [x for x in data['sets']['valid_pool']], transform=transform, crop_size=-1,
                                random_recs=False, triplet_dloss=triplet_dloss)
    test_pool_set = MSDataset3(data['inputs']['test_pool'], data['meta']['test_pool'], data['names']['test_pool'],
                               data['cats']['test_pool'], [x for x in data['batches']['test_pool']],
                               [x for x in data['sets']['test_pool']], transform=transform, crop_size=-1,
                               random_recs=False, triplet_dloss=triplet_dloss)
    valid_set = MSDataset3(data['inputs']['valid'], data['meta']['valid'], data['names']['valid'].to_numpy(),
                           data['cats']['valid'], [x for x in data['batches']['valid']],
                           [x for x in data['sets']['valid']], transform=transform, crop_size=-1, random_recs=False,
                           triplet_dloss=triplet_dloss)
    valid_set2 = MSDataset3(data['inputs']['valid'], data['meta']['valid'], data['names']['valid'].to_numpy(),
                            data['cats']['valid'], [x for x in data['batches']['valid']],
                            [x for x in data['sets']['valid']], transform=transform, crop_size=-1, random_recs=False,
                            triplet_dloss=triplet_dloss)
    test_set = MSDataset3(data['inputs']['test'], data['meta']['test'], data['names']['test'].to_numpy(),
                          data['cats']['test'], [x for x in data['batches']['test']], [x for x in data['sets']['test']],
                          transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss)
    test_set2 = MSDataset3(data['inputs']['test'][0][0], data['meta']['test'], data['names']['test'].to_numpy(),
                           data['cats']['test'], [x for x in data['batches']['test']],
                           [x for x in data['sets']['test']], transform=transform, crop_size=-1, random_recs=False,
                           triplet_dloss=triplet_dloss)

    loaders = {
        'train': DataLoader(train_set,
                            num_workers=0,
                            # shuffle=True,
                            sampler=WeightedRandomSampler(samples_weights['train'], len(samples_weights['train']),
                                                          replacement=True),
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=True),

        'test': DataLoader(test_set,
                           num_workers=0,
                           sampler=WeightedRandomSampler(samples_weights['test'], sum(samples_weights['test']),
                                                         replacement=False),
                           batch_size=bs,
                           pin_memory=False,
                           drop_last=False),
        'valid': DataLoader(valid_set,
                            num_workers=0,
                            sampler=WeightedRandomSampler(samples_weights['valid'], sum(samples_weights['valid']),
                                                          replacement=False),
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=False),

        'train_pool': DataLoader(train_pool_set,
                                 num_workers=0,
                                 shuffle=True,
                                 batch_size=bs,
                                 pin_memory=False,
                                 drop_last=False),

        'test_pool': DataLoader(test_pool_set,
                                num_workers=0,
                                shuffle=True,
                                batch_size=bs,
                                pin_memory=False,
                                drop_last=False),
        'valid_pool': DataLoader(valid_pool_set,
                                 num_workers=0,
                                 shuffle=True,
                                 batch_size=bs,
                                 pin_memory=False,
                                 drop_last=False),

        'test2': DataLoader(test_set2,
                            num_workers=0,
                            shuffle=True,
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=True),
        'valid2': DataLoader(valid_set2,
                             num_workers=0,
                             shuffle=True,
                             batch_size=bs,
                             pin_memory=False,
                             drop_last=True)
    }

    # TODO NOT MODIFIED FOR POOLS
    if ae is not None:
        valid_cats = []
        test_cats = []
        valid_names = []
        test_names = []
        ae.eval()
        classifier.eval()
        for i, batch in enumerate(loaders['valid']):
            # optimizer_ae.zero_grad()
            input, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            input[torch.isnan(input)] = 0
            input = input.to(device).float()
            enc, rec, _, kld = ae(input, domain, sampling=False)
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)
            valid_cats += [preds.detach().cpu().numpy().argmax(1)]
            valid_names += [names]
        for i, batch in enumerate(loaders['test']):
            # optimizer_ae.zero_grad()
            input, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            input[torch.isnan(input)] = 0
            input = input.to(device).float()
            # to_rec = to_rec.to(device).float()
            enc, rec, _, kld = ae(input, domain, sampling=False)
            # if self.one_model:
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)
            # else:
            #     preds = classifier(enc)
            test_cats += [preds.detach().cpu().numpy().argmax(1)]
            test_names += [names]

        valid_set2 = MSDataset3(data['inputs']['valid'], valid_names, np.concatenate(valid_cats),
                                [x for x in data['batches']['valid']], [x for x in data['sets']['valid']],
                                transform=transform, crop_size=-1, random_recs=random_recs, triplet_dloss=triplet_dloss)
        test_set2 = MSDataset3(data['inputs']['test'], test_names, np.concatenate(test_cats),
                               [x for x in data['batches']['test']], [x for x in data['sets']['test']],
                               transform=transform, crop_size=-1, random_recs=random_recs, triplet_dloss=triplet_dloss)
        loaders['valid2'] = DataLoader(valid_set2,
                                       num_workers=0,
                                       shuffle=True,
                                       batch_size=bs,
                                       pin_memory=False,
                                       drop_last=True)
        loaders['test2'] = DataLoader(test_set2,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=bs,
                                      pin_memory=False,
                                      drop_last=True)
        all_cats = np.concatenate(
            (data['cats']['train'], np.stack(valid_cats).reshape(-1), np.stack(test_cats).reshape(-1)))
        all_names = np.concatenate(
            (data['names']['train'], np.stack(valid_names).reshape(-1), np.stack(test_names).reshape(-1)))
        all_set = MSDataset3(data['inputs']['all'], all_names, all_cats, [x for x in data['time']['all']],
                             transform=transform, crop_size=-1, random_recs=random_recs, triplet_dloss=triplet_dloss)

    else:
        all_set = MSDataset3(data['inputs']['all'], data['meta']['all'], data['names']['all'], data['cats']['all'],
                             [x for x in data['batches']['all']], [x for x in data['sets']['all']], transform=transform,
                             crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss)
        all_set_pool = MSDataset3(data['inputs']['all_pool'], data['meta']['all_pool'], data['names']['all_pool'],
                                  data['cats']['all_pool'], [x for x in data['batches']['all_pool']],
                                  [x for x in data['sets']['all_pool']], transform=transform, crop_size=-1,
                                  random_recs=False, triplet_dloss=triplet_dloss)

    loaders['all'] = DataLoader(all_set,
                                num_workers=0,
                                shuffle=True,
                                batch_size=bs,
                                pin_memory=False,
                                drop_last=True)
    loaders['all_pool'] = DataLoader(all_set_pool,
                                     num_workers=0,
                                     shuffle=False,
                                     batch_size=bs,
                                     pin_memory=False,
                                     drop_last=False)

    return loaders


def get_images_loaders(data, random_recs, samples_weights, triplet_dloss, ae=None, classifier=None, bs=64,
                       device='cuda'):
    """

    Args:
        data:
        ae:
        classifier:

    Returns:

    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5),
        # torchvision.transforms.Normalize(np.mean(data['inputs']['train'].to_numpy().reshape(1, -1)), np.std(data['inputs']['train'].to_numpy().reshape(1, -1))),
    ])

    train_set = MSDataset4(data['inputs']['train'], data['meta']['train'], data['names']['train'],
                           data['cats']['train'],
                           data['batches']['train'],
                           transform=transform, crop_size=-1, random_recs=random_recs, triplet_dloss=triplet_dloss,
                           quantize=False, device=device)
    train_pool_set = MSDataset4(data['inputs']['train_pool'], data['meta']['train_pool'], data['names']['train_pool'],
                                data['cats']['train_pool'],
                                [x for x in data['batches']['train_pool']],
                                transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                                quantize=False, device=device)
    valid_pool_set = MSDataset4(data['inputs']['valid_pool'], data['meta']['valid_pool'], data['names']['valid_pool'],
                                data['cats']['valid_pool'],
                                [x for x in data['batches']['valid_pool']],
                                transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                                quantize=False, device=device)
    test_pool_set = MSDataset4(data['inputs']['test_pool'], data['meta']['test_pool'], data['names']['test_pool'],
                               data['cats']['test_pool'],
                               [x for x in data['batches']['test_pool']],
                               transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                               quantize=False, device=device)
    valid_set = MSDataset4(data['inputs']['valid'], data['meta']['valid'], data['names']['valid'],
                           data['cats']['valid'],
                           [x for x in data['batches']['valid']],
                           transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                           quantize=False, device=device)
    valid_set2 = MSDataset4(data['inputs']['valid'], data['meta']['valid'], data['names']['valid'],
                            data['cats']['valid'],
                            [x for x in data['batches']['valid']],
                            transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                            quantize=False, device=device)
    test_set = MSDataset4(data['inputs']['test'], data['meta']['test'], data['names']['test'],
                          data['cats']['test'],
                          [x for x in data['batches']['test']],
                          transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                          quantize=False, device=device)
    test_set2 = MSDataset4(data['inputs']['test'], data['meta']['test'], data['names']['test'],
                           data['cats']['test'],
                           [x for x in data['batches']['test']],
                           transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                           quantize=False, device=device)

    loaders = {
        'train': DataLoader(train_set,
                            num_workers=0,
                            # shuffle=True,
                            sampler=WeightedRandomSampler(samples_weights['train'], len(samples_weights['train']),
                                                          replacement=True),
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=True),

        'test': DataLoader(test_set,
                           num_workers=0,
                           sampler=WeightedRandomSampler(samples_weights['test'], sum(samples_weights['test']),
                                                         replacement=False),
                           batch_size=bs,
                           pin_memory=False,
                           drop_last=False),
        'valid': DataLoader(valid_set,
                            num_workers=0,
                            sampler=WeightedRandomSampler(samples_weights['valid'], sum(samples_weights['valid']),
                                                          replacement=False),
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=False),

        'train_pool': DataLoader(train_pool_set,
                                 num_workers=0,
                                 shuffle=True,
                                 batch_size=bs,
                                 pin_memory=False,
                                 drop_last=False),

        'test_pool': DataLoader(test_pool_set,
                                num_workers=0,
                                shuffle=True,
                                batch_size=bs,
                                pin_memory=False,
                                drop_last=False),
        'valid_pool': DataLoader(valid_pool_set,
                                 num_workers=0,
                                 shuffle=True,
                                 batch_size=bs,
                                 pin_memory=False,
                                 drop_last=False),

        'test2': DataLoader(test_set2,
                            num_workers=0,
                            shuffle=True,
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=True),
        'valid2': DataLoader(valid_set2,
                             num_workers=0,
                             shuffle=True,
                             batch_size=bs,
                             pin_memory=False,
                             drop_last=True)
    }

    # TODO NOT MODIFIED FOR POOLS
    if ae is not None:
        valid_cats = []
        test_cats = []
        valid_names = []
        test_names = []
        ae.eval()
        classifier.eval()
        for i, batch in enumerate(loaders['valid']):
            # optimizer_ae.zero_grad()
            input, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            input[torch.isnan(input)] = 0
            input = input.to(device).float()
            enc, rec, _, kld = ae(input, domain, sampling=False)
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)
            valid_cats += [preds.detach().cpu().numpy().argmax(1)]
            valid_names += [names]
        for i, batch in enumerate(loaders['test']):
            # optimizer_ae.zero_grad()
            input, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            input[torch.isnan(input)] = 0
            input = input.to(device).float()
            # to_rec = to_rec.to(device).float()
            enc, rec, _, kld = ae(input, domain, sampling=False)
            # if self.one_model:
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)
            # else:
            #     preds = classifier(enc)
            test_cats += [preds.detach().cpu().numpy().argmax(1)]
            test_names += [names]

        valid_set2 = MSDataset3(data['inputs']['valid'], valid_names, np.concatenate(valid_cats),
                                [x for x in data['batches']['valid']], transform=transform, crop_size=-1,
                                random_recs=random_recs, triplet_dloss=triplet_dloss)
        test_set2 = MSDataset3(data['inputs']['test'], test_names, np.concatenate(test_cats),
                               [x for x in data['batches']['test']], transform=transform, crop_size=-1,
                               random_recs=random_recs, triplet_dloss=triplet_dloss)
        loaders['valid2'] = DataLoader(valid_set2,
                                       num_workers=0,
                                       shuffle=True,
                                       batch_size=bs,
                                       pin_memory=False,
                                       drop_last=True)
        loaders['test2'] = DataLoader(test_set2,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=bs,
                                      pin_memory=False,
                                      drop_last=True)
        all_cats = np.concatenate(
            (data['cats']['train'], np.stack(valid_cats).reshape(-1), np.stack(test_cats).reshape(-1)))
        all_names = np.concatenate(
            (data['names']['train'], np.stack(valid_names).reshape(-1), np.stack(test_names).reshape(-1)))
        all_set = MSDataset3(data['inputs']['all'], all_names, all_cats, [x for x in data['time']['all']],
                             transform=transform, crop_size=-1, random_recs=random_recs, triplet_dloss=triplet_dloss)

    else:
        all_set = MSDataset3(data['inputs']['all'], data['meta']['all'], data['names']['all'], data['cats']['all'],
                             [x for x in data['batches']['all']], transform=transform, crop_size=-1, random_recs=False,
                             triplet_dloss=triplet_dloss)
        all_set_pool = MSDataset3(data['inputs']['all_pool'], data['meta']['all_pool'], data['names']['all_pool'],
                                  data['cats']['all_pool'], [x for x in data['batches']['all_pool']],
                                  transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss)

    loaders['all'] = DataLoader(all_set,
                                num_workers=0,
                                shuffle=True,
                                batch_size=bs,
                                pin_memory=False,
                                drop_last=True)
    loaders['all_pool'] = DataLoader(all_set_pool,
                                     num_workers=0,
                                     shuffle=False,
                                     batch_size=bs,
                                     pin_memory=False,
                                     drop_last=False)

    return loaders


def get_images_loaders_no_pool(data, random_recs, samples_weights, triplet_dloss, ae=None, classifier=None, bs=64,
                               device='cuda'):
    """

    Args:
        data:
        ae:
        classifier:

    Returns:

    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32),
        transforms.Grayscale(),
        # transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(90, 90))])),
        # transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(180, 180))])),
        # transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(270, 270))])),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        # transforms.RandomApply(nn.ModuleList([
        #     transforms.RandomAffine(
        #         degrees=10,
        #         translate=(.1, .1),
        #         # scale=(0.9, 1.1),
        #         shear=(.01, .01),
        #         interpolation=transforms.InterpolationMode.BILINEAR
        #     )
        # ]), p=0.5),
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 0.03))
        #     ]), p=0.5),
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.RandomResizedCrop(
        #             size=32,
        #             scale=(0.8, 1.),
        #             ratio=(0.8, 1.2)
        #         )
        #     ]), p=0.5),
        # torchvision.transforms.Normalize(0.5, 0.5),
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize(32),
        # torchvision.transforms.Normalize(0.5, 0.5),
    ])

    train_set = MSDataset5(data['inputs']['train'], data['meta']['train'], data['names']['train'],
                           data['cats']['train'],
                           data['batches']['train'],
                           sets=data['sets']['train'],
                           transform=transform_train, crop_size=-1, random_recs=random_recs,
                           triplet_dloss=triplet_dloss,
                           quantize=False, device=device)
    valid_set = MSDataset5(data['inputs']['valid'], data['meta']['valid'], data['names']['valid'],
                           data['cats']['valid'],
                           [x for x in data['batches']['valid']],
                           sets=data['sets']['valid'],
                           transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                           quantize=False, device=device)
    valid_set2 = MSDataset5(data['inputs']['valid'], data['meta']['valid'], data['names']['valid'],
                            data['cats']['valid'],
                            [x for x in data['batches']['valid']],
                            sets=data['sets']['valid'],
                            transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                            quantize=False, device=device)
    test_set = MSDataset5(data['inputs']['test'], data['meta']['test'], data['names']['test'],
                          data['cats']['test'],
                          [x for x in data['batches']['test']],
                          sets=data['sets']['test'],
                          transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                          quantize=False, device=device)
    test_set2 = MSDataset5(data['inputs']['test'], data['meta']['test'], data['names']['test'],
                           data['cats']['test'],
                           [x for x in data['batches']['test']],
                           sets=data['sets']['test'],
                           transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss,
                           quantize=False, device=device)

    loaders = {
        'train': DataLoader(train_set,
                            num_workers=0,
                            shuffle=True,
                            # sampler=WeightedRandomSampler(samples_weights['train'], len(samples_weights['train']),
                            #                               replacement=True),
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=True),

        'test': DataLoader(test_set,
                           num_workers=0,
                           # sampler=WeightedRandomSampler(samples_weights['test'], sum(samples_weights['test']),
                           #                               replacement=False),
                           batch_size=bs,
                           pin_memory=False,
                           drop_last=False),
        'valid': DataLoader(valid_set,
                            num_workers=0,
                            # sampler=WeightedRandomSampler(samples_weights['valid'], sum(samples_weights['valid']),
                            #                               replacement=False),
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=False),

        'test2': DataLoader(test_set2,
                            num_workers=0,
                            shuffle=True,
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=True),
        'valid2': DataLoader(valid_set2,
                             num_workers=0,
                             shuffle=True,
                             batch_size=bs,
                             pin_memory=False,
                             drop_last=True)
    }

    # TODO NOT MODIFIED FOR POOLS
    if ae is not None:
        valid_cats = []
        test_cats = []
        valid_names = []
        test_names = []
        ae.eval()
        classifier.eval()
        for i, batch in enumerate(loaders['valid']):
            # optimizer_ae.zero_grad()
            input, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            input[torch.isnan(input)] = 0
            input = input.to(device).float()
            enc, rec, _, kld = ae(input, domain, sampling=False)
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)
            valid_cats += [preds.detach().cpu().numpy().argmax(1)]
            valid_names += [names]
        for i, batch in enumerate(loaders['test']):
            # optimizer_ae.zero_grad()
            input, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            input[torch.isnan(input)] = 0
            input = input.to(device).float()
            # to_rec = to_rec.to(device).float()
            enc, rec, _, kld = ae(input, domain, sampling=False)
            # if self.one_model:
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)
            # else:
            #     preds = classifier(enc)
            test_cats += [preds.detach().cpu().numpy().argmax(1)]
            test_names += [names]

        valid_set2 = MSDataset5(data['inputs']['valid'], valid_names, np.concatenate(valid_cats),
                                [x for x in data['batches']['valid']],
                                sets=data['sets']['valid'],
                                transform=transform, crop_size=-1, random_recs=random_recs, triplet_dloss=triplet_dloss,
                                quantize=False, device=device)
        test_set2 = MSDataset5(data['inputs']['test'], test_names, np.concatenate(test_cats),
                               [x for x in data['batches']['test']],
                               sets=data['sets']['test'],
                               transform=transform, crop_size=-1, random_recs=random_recs, triplet_dloss=triplet_dloss,
                               quantize=False, device=device)
        loaders['valid2'] = DataLoader(valid_set2,
                                       num_workers=0,
                                       shuffle=True,
                                       batch_size=bs,
                                       pin_memory=False,
                                       drop_last=True)
        loaders['test2'] = DataLoader(test_set2,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=bs,
                                      pin_memory=False,
                                      drop_last=True)
        all_cats = np.concatenate(
            (data['cats']['train'], np.stack(valid_cats).reshape(-1), np.stack(test_cats).reshape(-1)))
        all_names = np.concatenate(
            (data['names']['train'], np.stack(valid_names).reshape(-1), np.stack(test_names).reshape(-1)))
        all_set = MSDataset5(data['inputs']['all'], all_names, all_cats,
                             [x for x in data['time']['all']], transform=transform_train, crop_size=-1,
                             random_recs=random_recs, quantize=False, triplet_dloss=triplet_dloss, device=device)

    else:
        all_set = MSDataset5(data['inputs']['all'], data['meta']['all'], data['names']['all'], data['cats']['all'],
                             [x for x in data['batches']['all']],
                             sets=data['sets']['all'],
                             transform=transform_train, crop_size=-1,
                             random_recs=False, quantize=False, triplet_dloss=triplet_dloss, device=device)

    loaders['all'] = DataLoader(all_set,
                                num_workers=0,
                                shuffle=True,
                                batch_size=bs,
                                pin_memory=False,
                                drop_last=True)

    return loaders


def get_loaders_no_pool(data, random_recs, samples_weights, triplet_dloss, ae=None, classifier=None, bs=64,
                        device='cuda'):
    """

    Args:
        data:
        ae:
        classifier:

    Returns:

    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(0.5, 0.5),
    ])

    train_set = MSDataset3(data['inputs']['train'], data['meta']['train'], data['names']['train'],
                           data['cats']['train'], [x for x in data['batches']['train']], sets=data['sets']['all'],
                           transform=transform, crop_size=-1, random_recs=random_recs, triplet_dloss=triplet_dloss)
    valid_set = MSDataset3(data['inputs']['valid'], data['meta']['valid'], data['names']['valid'],
                           data['cats']['valid'], [x for x in data['batches']['valid']], sets=data['sets']['valid'],
                           transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss)
    valid_set2 = MSDataset3(data['inputs']['valid'], data['meta']['valid'], data['names']['valid'],
                            data['cats']['valid'], [x for x in data['batches']['valid']], sets=data['sets']['valid'],
                            transform=transform, crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss)
    test_set = MSDataset3(data['inputs']['test'], data['meta']['test'], data['names']['test'], data['cats']['test'],
                          [x for x in data['batches']['test']], sets=data['sets']['test'], transform=transform,
                          crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss)
    test_set2 = MSDataset3(data['inputs']['test'], data['meta']['test'], data['names']['test'], data['cats']['test'],
                           [x for x in data['batches']['test']], sets=data['sets']['test'], transform=transform,
                           crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss)

    loaders = {
        'train': DataLoader(train_set,
                            num_workers=0,
                            # shuffle=True,
                            sampler=WeightedRandomSampler(samples_weights['train'], len(samples_weights['train']),
                                                          replacement=True),
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=True),

        'test': DataLoader(test_set,
                           num_workers=0,
                           sampler=WeightedRandomSampler(samples_weights['test'], sum(samples_weights['test']),
                                                         replacement=False),
                           batch_size=bs,
                           pin_memory=False,
                           drop_last=False),
        'valid': DataLoader(valid_set,
                            num_workers=0,
                            sampler=WeightedRandomSampler(samples_weights['valid'], sum(samples_weights['valid']),
                                                          replacement=False),
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=False),

        'test2': DataLoader(test_set2,
                            num_workers=0,
                            shuffle=True,
                            batch_size=bs,
                            pin_memory=False,
                            drop_last=True),
        'valid2': DataLoader(valid_set2,
                             num_workers=0,
                             shuffle=True,
                             batch_size=bs,
                             pin_memory=False,
                             drop_last=True)
    }

    # TODO NOT MODIFIED FOR POOLS
    if ae is not None:
        valid_cats = []
        test_cats = []
        valid_names = []
        test_names = []
        ae.eval()
        classifier.eval()
        for i, batch in enumerate(loaders['valid']):
            # optimizer_ae.zero_grad()
            input, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            input[torch.isnan(input)] = 0
            input = input.to(device).float()
            enc, rec, _, kld = ae(input, domain, sampling=False)
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)
            valid_cats += [preds.detach().cpu().numpy().argmax(1)]
            valid_names += [names]
        for i, batch in enumerate(loaders['test']):
            # optimizer_ae.zero_grad()
            input, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            input[torch.isnan(input)] = 0
            input = input.to(device).float()
            # to_rec = to_rec.to(device).float()
            enc, rec, _, kld = ae(input, domain, sampling=False)
            # if self.one_model:
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)
            # else:
            #     preds = classifier(enc)
            test_cats += [preds.detach().cpu().numpy().argmax(1)]
            test_names += [names]

        valid_set2 = MSDataset3(data['inputs']['valid'], valid_names, np.concatenate(valid_cats),
                                [x for x in data['batches']['valid']], sets=data['sets']['valid'], transform=transform,
                                crop_size=-1, random_recs=random_recs, triplet_dloss=triplet_dloss)
        test_set2 = MSDataset3(data['inputs']['test'], test_names, np.concatenate(test_cats),
                               [x for x in data['batches']['test']], sets=data['sets']['test'], transform=transform,
                               crop_size=-1, random_recs=random_recs, triplet_dloss=triplet_dloss)
        loaders['valid2'] = DataLoader(valid_set2,
                                       num_workers=0,
                                       shuffle=True,
                                       batch_size=bs,
                                       pin_memory=False,
                                       drop_last=True)
        loaders['test2'] = DataLoader(test_set2,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=bs,
                                      pin_memory=False,
                                      drop_last=True)
        all_cats = np.concatenate(
            (data['cats']['train'], np.stack(valid_cats).reshape(-1), np.stack(test_cats).reshape(-1)))
        all_names = np.concatenate(
            (data['names']['train'], np.stack(valid_names).reshape(-1), np.stack(test_names).reshape(-1)))
        all_set = MSDataset3(data['inputs']['all'], all_names, all_cats, [x for x in data['time']['all']],
                             sets=data['sets']['all'], transform=transform, crop_size=-1, random_recs=random_recs,
                             triplet_dloss=triplet_dloss)

    else:
        all_set = MSDataset3(data['inputs']['all'], data['meta']['all'], data['names']['all'], data['cats']['all'],
                             [x for x in data['batches']['all']], sets=data['sets']['all'], transform=transform,
                             crop_size=-1, random_recs=False, triplet_dloss=triplet_dloss)

    loaders['all'] = DataLoader(all_set,
                                num_workers=0,
                                shuffle=True,
                                batch_size=bs,
                                pin_memory=False,
                                drop_last=True)

    return loaders


class MSCSV:
    def __init__(self, path, scaler, new_size=32, test=False, resize=True):
        self.path = path
        self.resize = resize
        self.new_size = new_size
        self.scaler = scaler
        self.fnames = []
        if not test:
            self.fnames.extend(os.listdir(f"{path}"))
        else:
            tmp = os.listdir(f"{path}")
            np.random.shuffle(tmp)
            self.fnames = tmp[:57]

    def process(self, i):
        fname = self.fnames[i]
        b_list = ["kox", "sau", "blk", "pae", "sep"]
        batch = fname.split('_')[0]
        label = fname.split('_')[1]
        plate = fname.split('_')[2]
        print(f"Processing sample #{i}: {fname}")
        mat_data = read_csv(f"{self.path}/{fname}")
        if self.scaler == 'binarize':
            mat_data[mat_data.values > 0] = 1
        elif 'efd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif 'ewd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif 'kmd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)

        elif 'cut' in self.scaler:
            n_bins = int(self.scaler.split('_')[1])
            mat_data = pd.DataFrame(
                np.stack([pd.cut(row, n_bins, labels=False, duplicates='drop', include_lowest=True) for row in
                          mat_data.values.T]).T
            )
            mat_data /= mat_data.max()
        elif 'discretizeq' in self.scaler:
            n_bins = int(self.scaler.split('_')[1])
            mat_data = pd.DataFrame(
                np.stack([pd.qcut(row, n_bins, labels=False, duplicates='drop') for row in mat_data.values.T]).T
            )

            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif self.scaler == 'l2':
            mat_data = Normalizer().fit_transform(mat_data)
        elif self.scaler == 'l1':
            mat_data = Normalizer('l1').fit_transform(mat_data)
        elif self.scaler == 'minmax':
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif self.scaler == 'max':
            mat_data = Normalizer('max').fit_transform(mat_data)
        elif self.scaler == 'maxmax':
            mat_data /= mat_data.max().max()

        if self.resize:
            try:
                mat_data = transforms.Resize((self.new_size, self.new_size))(
                torch.Tensor(mat_data.values).unsqueeze(0)).squeeze().detach().cpu().numpy()
            except:
                mat_data = transforms.Resize((self.new_size, self.new_size))(
                torch.Tensor(mat_data).unsqueeze(0)).squeeze().detach().cpu().numpy()

        return mat_data.astype('float'), label, batch, plate, fname.split('.csv')[0]

    def __len__(self):
        return len(self.fnames)


class MS2CSV:
    def __init__(self, path, scaler, new_size=32, test=False, resize=True):
        self.path = path
        self.resize = resize
        self.new_size = new_size
        self.scaler = scaler
        self.fnames = []
        if not test:
            self.fnames.extend(os.listdir(f"{path}"))
        else:
            tmp = os.listdir(f"{path}")
            np.random.shuffle(tmp)
            self.fnames = tmp[:57]

    def process(self, i):
        fname = self.fnames[i]
        batch = fname.split('_')[0]
        label = fname.split('_')[1]
        plate = fname.split('_')[2]
        print(f"Processing sample #{i}: {fname}")
        mat_datas = []
        for f in os.listdir(f"{self.path}/{fname}"):
            mat_data = read_csv(f"{self.path}/{fname}/{f}")
            if self.scaler == 'binarize':
                mat_data[mat_data.values > 0] = 1
            elif 'efd' in self.scaler:
                from sklearn.preprocessing import KBinsDiscretizer
                n_bins = int(self.scaler.split('_')[1])
                mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit_transform(mat_data)
                mat_data = MinMaxScaler().fit_transform(mat_data)
            elif 'ewd' in self.scaler:
                from sklearn.preprocessing import KBinsDiscretizer
                n_bins = int(self.scaler.split('_')[1])
                mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit_transform(mat_data)
                mat_data = MinMaxScaler().fit_transform(mat_data)
            elif 'kmd' in self.scaler:
                from sklearn.preprocessing import KBinsDiscretizer
                n_bins = int(self.scaler.split('_')[1])
                mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans').fit_transform(mat_data)
                mat_data = MinMaxScaler().fit_transform(mat_data)

            elif 'cut' in self.scaler:
                n_bins = int(self.scaler.split('_')[1])
                mat_data = pd.DataFrame(
                    np.stack([pd.cut(row, n_bins, labels=False, duplicates='drop', include_lowest=True) for row in
                              mat_data.values.T]).T
                )
                mat_data /= mat_data.max()
            elif 'discretizeq' in self.scaler:
                n_bins = int(self.scaler.split('_')[1])
                mat_data = pd.DataFrame(
                    np.stack([pd.qcut(row, n_bins, labels=False, duplicates='drop') for row in mat_data.values.T]).T
                )

                mat_data = MinMaxScaler().fit_transform(mat_data)
            elif self.scaler == 'l2':
                mat_data = Normalizer().fit_transform(mat_data)
            elif self.scaler == 'l1':
                mat_data = Normalizer('l1').fit_transform(mat_data)
            elif self.scaler == 'minmax':
                mat_data = MinMaxScaler().fit_transform(mat_data)
            elif self.scaler == 'max':
                mat_data = Normalizer('max').fit_transform(mat_data)
            elif self.scaler == 'maxmax':
                mat_data /= mat_data.max().max()

            if self.resize:
                try:
                    mat_data = transforms.Resize((self.new_size, self.new_size))(
                    torch.Tensor(mat_data.values).unsqueeze(0)).squeeze().detach().cpu().numpy()
                except:
                    mat_data = transforms.Resize((self.new_size, self.new_size))(
                    torch.Tensor(mat_data).unsqueeze(0)).squeeze().detach().cpu().numpy()
            mat_datas += [mat_data]
        # try:
        mat_data = np.stack(mat_datas, 0)
        # except:
        #     pass
        return mat_data.astype('float'), label, batch, plate, fname.split('.csv')[0]

    def __len__(self):
        return len(self.fnames)


class MSCSV2:
    def __init__(self, path, scaler, batches, resize=False, test=False):
        self.batches = batches
        self.path = path
        self.resize = resize
        self.scaler = scaler
        self.fnames = []
        if not test:
            self.fnames.extend(os.listdir(f"{path}"))
        else:
            tmp = os.listdir(f"{path}")
            np.random.shuffle(tmp)
            self.fnames = tmp[:57]
        tmp = []
        for fname in self.fnames:
            batch = [i for i, b in enumerate(self.batches) if b == fname.split('_')[0]]
            if len(batch) > 0:
                tmp += [fname]
        self.fnames = tmp

    def process(self, i):
        fname = self.fnames[i]
        if 'l' in fname.split('_')[2]:
            low = 1
        else:
            low = 0

        batch = [i for i, b in enumerate(self.batches) if b == fname.split('_')[0]][0]
        label = fname.split('_')[1]
        print(f"Processing sample #{i}: {fname}")
        mat_data = read_csv(f"{self.path}/{fname}").values
        if self.scaler == 'binarize':
            mat_data[mat_data > 0] = 1
        elif 'efd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif 'ewd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif 'kmd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)

        elif 'cut' in self.scaler:
            n_bins = int(self.scaler.split('_')[1])
            mat_data = pd.DataFrame(
                np.stack([pd.cut(row, n_bins, labels=False, duplicates='drop', include_lowest=True) for row in
                          mat_data.values.T]).T
            )
            mat_data /= mat_data.max()
        elif 'discretizeq' in self.scaler:
            n_bins = int(self.scaler.split('_')[1])
            mat_data = pd.DataFrame(
                np.stack([pd.qcut(row, n_bins, labels=False, duplicates='drop') for row in mat_data.values.T]).T
            )

            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif self.scaler == 'l2':
            mat_data = Normalizer().fit_transform(mat_data)
        elif self.scaler == 'l1':
            mat_data = Normalizer('l1').fit_transform(mat_data)
        elif self.scaler == 'minmax':
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif self.scaler == 'max':
            mat_data = Normalizer('max').fit_transform(mat_data)
        elif self.scaler == 'maxmax':
            mat_data /= mat_data.max().max()

        if self.resize:
            mat_data = transforms.Resize((299, 299))(
                torch.Tensor(mat_data).unsqueeze(0)).squeeze().detach().cpu().numpy()

        return mat_data.astype('float'), label, low, batch, fname

    def __len__(self):
        return len(self.fnames)


