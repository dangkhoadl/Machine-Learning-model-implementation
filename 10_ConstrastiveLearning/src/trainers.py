import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from src.utils import plot_features, compute_metrics


class Trainer(nn.Module):
    def __init__(self,
            model, optimizer, criterion,
            main_scheduler, warmup_scheduler,
            train_dataset, val_dataset, labels_dict,
            batch_size, num_workers,
            exp_name, device='cuda:0', mode='pretrain'):
        super().__init__()
        assert mode in ['pretrain', 'downstream']

        # Models and criterion
        self._model = model
        self._criterion = criterion

        # Optimizers and schedulers
        self._optimizer = optimizer
        self._warmup_scheduler = warmup_scheduler
        self._main_scheduler = main_scheduler

        # Datasets
        self._val_dataset = val_dataset
        self._labels_dict = labels_dict

        # Dataloaders
        self._train_loader = DataLoader(train_dataset,
            batch_size=batch_size, num_workers=num_workers,
            drop_last=True,
            shuffle=True)

        self._eval_loader = DataLoader(val_dataset,
            batch_size=batch_size, num_workers=num_workers,
            drop_last=True,
            shuffle=False)

        ## dirs + paths
        self._exp_name = exp_name
        # chkpt
        chkpt_dir = f"exp/{self._exp_name}/chkpts"
        self._chkpt_path =  os.path.join(
            chkpt_dir, f"best_{mode}ed.pt")
        if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
        # tsne plots
        self._plot_dir = f"exp/{self._exp_name}/plotting"
        if not os.path.exists(self._plot_dir):
            os.makedirs(self._plot_dir)

        # Params
        self._device = device
        self._step = 0
        self._best_eval_loss = np.inf
        self._start_epoch = 0

        # Logs
        self._train_log = {
            'epoch': [],
            'step': [],
            'train_loss': [],
            'lr': []
        }
        self._eval_log = {
            'epoch': [],
            'step': [],
            'eval_loss': [],
        }
        self._train_log_path = f"exp/{self._exp_name}/{mode}ed_trainlog.csv"
        self._eval_log_path = f"exp/{self._exp_name}/{mode}ed_evallog.csv"

    def save_chkpt(self, current_epoch):
        torch.save({
            'epoch': current_epoch,
            'step': self._step,
            'best_eval_loss': self._best_eval_loss,

            'train_log': self._train_log,
            'eval_log': self._eval_log,

            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'warmup_scheduler':self._warmup_scheduler.state_dict(),
            'main_scheduler':self._main_scheduler.state_dict()},
        self._chkpt_path)

    def load_chkpt(self):
        if not os.path.exists(self._chkpt_path):
            raise ValueError("Checkpoint file does not exist")
        chkpt = torch.load(self._chkpt_path)

        self._step = chkpt['step']
        self._best_eval_loss = chkpt['best_eval_loss']
        self._start_epoch = chkpt['epoch']

        self._train_log = chkpt['train_log']
        self._eval_log = chkpt['eval_log']

        self._model.load_state_dict(chkpt['model'])
        self._optimizer.load_state_dict(chkpt['optimizer'])
        self._warmup_scheduler.load_state_dict(chkpt['warmup_scheduler'])
        self._main_scheduler.load_state_dict(chkpt['main_scheduler'])


    def train_epoch(self, epoch): pass

    def eval_epoch(self, epoch, plot_feats): pass

    def train(self, num_epochs, eval_every_epochs=10, resume_from_chkpt=False, plot_feats=False):
        if resume_from_chkpt:
            self.load_chkpt()

        for epoch in tqdm(range(self._start_epoch, num_epochs), desc='Epoch'):
            ## Train epoch
            self.train_epoch(epoch)

            ## Eval epoch
            if (epoch+1) % eval_every_epochs == 0 or epoch == num_epochs-1:
                self.eval_epoch(epoch, plot_feats)


class PretrainTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_epoch(self, epoch):
        tr_loss_epoch = 0

        self._model.train()
        for _, batch in enumerate(tqdm(self._train_loader, desc='Minibatch')):
            # Load batch
            x1, x2 = batch['x1'], batch['x2']
            x1, x2 = x1.to(self._device).float(), x2.to(self._device).float()
            # Supervised contrastive learning
            if self._exp_name == 'SupCon':
                labels = batch['label'].to(self._device).long()
            else:
                labels = None

            # positive pair, with encoding
            z1 = self._model(x1) # (m, feat_dim)
            z2 = self._model(x2) # (m, feat_dim)

            # L2 Normalization - prevent exploding gradients
            z1 = F.normalize(z1, p=2, dim=-1)
            z2 = F.normalize(z2, p=2, dim=-1)

            # feats.shape = (m, 2, feat_dime)
            feats = torch.stack([z1, z2], dim=1)

            # Update loss
            self._optimizer.zero_grad()
            if self._exp_name == 'SupCon':
                loss = self._criterion(feats, labels)
            else:
                loss = self._criterion(feats)
            loss.backward()
            tr_loss_epoch += loss.item()

            # optimizer step
            self._optimizer.step()
            self._step += 1

        # Normalize loss
        train_loss = tr_loss_epoch / len(self._train_loader)

        # Scheduler step
        if epoch < 10:
            self._warmup_scheduler.step()
        if epoch >= 10:
            self._main_scheduler.step()

        # Log
        lr = self._optimizer.param_groups[0]["lr"]
        self._train_log['epoch'].append(epoch)
        self._train_log['step'].append(self._step)
        self._train_log['train_loss'].append(train_loss)
        self._train_log['lr'].append(lr)
        df = pd.DataFrame(self._train_log)
        df.to_csv(self._train_log_path, index=False)

        # Reshuffle train_dset
        self._train_loader.dataset.on_epoch_end()

    def eval_epoch(self, epoch, plot_feats):
        val_loss_epoch = 0

        self._model.eval()
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self._eval_loader, desc='Evaluation')):
                # Load batch
                x1, x2 = batch['x1'], batch['x2']
                x1, x2 = x1.to(self._device).float(), x2.to(self._device).float()

                # positive pair, with encoding
                z1 = self._model(x1) # (m, feat_dim)
                z2 = self._model(x2) # (m, feat_dim)

                # feats.shape = (m, 2, feat_dime)
                feats = torch.stack([z1, z2], dim=1)

                # update loss
                loss = self._criterion(feats)
                val_loss_epoch += loss.item()

        # Normalize loss
        eval_loss = val_loss_epoch / len(self._eval_loader)

        # Save best checkpoint
        if self._best_eval_loss > eval_loss:
            self._best_eval_loss = eval_loss
            self.save_chkpt(epoch)

            # Plot tsne
            if plot_feats:
                plot_path = os.path.join(
                    self._plot_dir, f"tsne_{epoch}.png")
                plot_features(
                    model=self._model.pretrained,
                    plot_dataset=self._val_dataset, labels_dict=self._labels_dict,
                    save_path=plot_path)

        # Log
        self._eval_log['epoch'].append(epoch)
        self._eval_log['step'].append(self._step)
        self._eval_log['eval_loss'].append(eval_loss / len(self._eval_loader))
        df = pd.DataFrame(self._eval_log)
        df.to_csv(self._eval_log_path, index=False)


class DownstreamTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_log = {
            'epoch': [],
            'step': [],
            'eval_loss': [],
            'accuracy': [],
        }

    def train_epoch(self, epoch):
        tr_loss_epoch = 0

        self._model.train()
        for _, batch in enumerate(tqdm(self._train_loader, desc='Minibatch')):
            # Load batch
            imgs = batch['img'].to(self._device, dtype=torch.float)
            labels = batch['label'].to(self._device, dtype=torch.long)

            # positive pair, with encoding
            preds = self._model(imgs)

            # update loss
            self._optimizer.zero_grad()
            loss = self._criterion(preds, labels)
            loss.backward()
            tr_loss_epoch += loss.item()

            # optimizer step
            self._optimizer.step()
            self._step += 1

        # Normalize loss
        train_loss = tr_loss_epoch / len(self._train_loader)

        # Scheduler step
        self._main_scheduler.step()

        # Log
        lr = self._optimizer.param_groups[0]["lr"]
        self._train_log['epoch'].append(epoch)
        self._train_log['step'].append(self._step)
        self._train_log['train_loss'].append(train_loss)
        self._train_log['lr'].append(lr)
        df = pd.DataFrame(self._train_log)
        df.to_csv(self._train_log_path, index=False)

        # Reshuffle train_dset
        self._train_loader.dataset.on_epoch_end()

    def eval_epoch(self, epoch, plot_feats=False):
        val_loss_epoch = 0
        preds_list, labels_list = [], []

        self._model.eval()
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self._eval_loader, desc='Evaluation')):
                # Load batch
                imgs = batch['img'].to(self._device, dtype=torch.float)
                labels = batch['label'].to(self._device, dtype=torch.long)
                labels_list.append(labels.cpu().numpy())

                # positive pair, with encoding
                preds = self._model(imgs)
                preds_list.append(preds.cpu().numpy())

                # update loss
                loss = self._criterion(preds, labels)
                val_loss_epoch += loss.item()

        # Normalize loss
        eval_loss = val_loss_epoch / len(self._eval_loader)

        # Save best checkpoint
        if self._best_eval_loss > eval_loss:
            self._best_eval_loss = eval_loss
            self.save_chkpt(epoch)

            # Plot tsne
            if plot_feats:
                plot_path = os.path.join(
                    self._plot_dir, f"tsne_{epoch}.png")
                plot_features(
                    model=self._model.pretrained,
                    plot_dataset=self._val_dataset, labels_dict=self._labels_dict,
                    save_path=plot_path)

        # Log
        self._eval_log['epoch'].append(epoch)
        self._eval_log['step'].append(self._step)
        self._eval_log['eval_loss'].append(eval_loss / len(self._eval_loader))

        stats = compute_metrics(
            preds=np.concatenate(preds_list, axis=0),
            labels=np.concatenate(labels_list, axis=0))
        self._eval_log['accuracy'].append(stats['accuracy'])
        df = pd.DataFrame(self._eval_log)
        df.to_csv(self._eval_log_path, index=False)