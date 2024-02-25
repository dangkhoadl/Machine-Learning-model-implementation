# -*- coding: utf-8 -*-
import sys, os
import yaml
import json
from glob import glob
from hyperpyyaml import load_hyperpyyaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch

from src.models import \
    Resnet_SCL_Config, Resnet_SCL
from src.datasets import lab_dict, unpickle, Pretrained_Dset
from src.optimizers import LARS

from transformers import Trainer, TrainerCallback


def parse_arguments(args):
    yaml_fpath, *override_args = args

    # Parse overrides
    assert len(override_args) % 2 == 0
    overrides = dict()
    for i in np.arange(0, len(override_args), 2):
        param, val = override_args[i].lstrip('--'), override_args[i+1]
        overrides[param] = val

    # Parse config.yaml
    with open(yaml_fpath, 'r+') as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    return hparams


def load_training_data():
    # Load data
    images = np.array([], dtype=np.uint8).reshape((0, 3072))
    labels = np.array([])

    for file in glob("DATA/train_batch_*"):
        data_dict = unpickle(file)

        images = np.append(images, data_dict[b'data'], axis=0)
        labels = np.append(labels,data_dict[b'labels'])

    images = images.reshape((-1,3,32,32)).astype(np.float)

    # train, eval split
    train_images, val_images = images[:40000], images[40000:]
    train_labels, val_labels = labels[:40000], labels[40000:]

    # Extract mean, std
    MEAN = np.mean(train_images/255.0, axis=(0,2,3), keepdims=True)
    STD = np.std(train_images/255.0, axis=(0,2,3), keepdims=True)

    return train_images, train_labels, val_images, val_labels, MEAN, STD


def pretrain(hparams):
    train_images, train_labels, val_images, val_labels, MEAN, STD = load_training_data()

    # Load Datasets
    train_dset = Pretrained_Dset(train_images, train_labels,
            mode='train',
            mean=MEAN, std=STD)
    eval_dset = Pretrained_Dset(val_images, val_labels,
            mode='valid',
            mean=MEAN, std=STD)

    # Load model
    config = Resnet_SCL_Config(
        num_classes=len(lab_dict),
        mode='pretrain')
    model = Resnet_SCL(
        config=config)

    # optimizer, scheduler, loss function
    optimizer = LARS(
        [params for params in model.parameters() if params.requires_grad],
        lr=hparams['pretrain_lr'],
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch : (epoch+1)/10.0,
        verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=500, eta_min=0.05, last_epoch=-1, verbose=True)

    class Custom_Trainer(Trainer): pass

    # Eval metrics: TSNE projection
    plot_dir = f"{hparams['exp_dir']}/images"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    def compute_metrics(eval_pred, average='macro'):
        assert average in [None, 'micro', 'macro', 'weighted']

        # Extract logits and labels
        projected_embeddings = eval_pred.predictions
        labels = eval_pred.label_ids

        ## Plot projected embeddings
        # tsne: reduce dimensionality for easy plotting
        tsne_model = TSNE(n_components=2, perplexity=50)
        feats_2D = tsne_model.fit_transform(projected_embeddings)

        # Plot tsne feats
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

        for idx, class_name in sorted(lab_dict.items(), key=lambda x: x[0]):
            ax.scatter(
                x=feats_2D[labels==idx, 1],
                y=feats_2D[labels==idx, 0],
                s=5, label=f"Class {idx}: {class_name}")
        ax.legend(
            loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        return {"figure": fig}

    class PrinterCallback(TrainerCallback):
        def __write_log(self, state):
            train_log, eval_log = [], []

            for e in state.log_history:
                e_keys = set(e)
                if "loss" in e_keys: train_log.append(e)
                elif "eval_loss" in e_keys: eval_log.append(e)
                elif "train_runtime" in e_keys:
                    with open(f"{hparams['pretrain_exp_dir']}/trainer_info.json", 'w+', encoding='utf-8') as fin:
                        json.dump(e, fin, ensure_ascii=False, indent=4)

            if train_log != []:
                train_log_df = pd.DataFrame.from_dict(train_log) \
                    .sort_values("step", ascending=True) \
                    .reset_index(drop=True)
                train_log_df.to_csv(f"{hparams['pretrain_exp_dir']}/log_trainset.csv", index=False)

            if eval_log != []:
                eval_log_df = pd.DataFrame.from_dict(eval_log) \
                    .sort_values("step", ascending=True) \
                    .reset_index(drop=True)
                eval_log_df.to_csv(f"{hparams['pretrain_exp_dir']}/log_evalset.csv", index=False)

        def on_evaluate(self, args, state, control, logs=None, **kwargs):
            '''Write log after every eval round'''
            # Pop fig + save
            fig = state.log_history[-1].pop('eval_figure')
            last_step = state.log_history[-1]['step']

            fig.savefig(f"{plot_dir}/tsne_feats_{last_step}.png",
                dpi=100, bbox_inches='tight', facecolor='white')

            # Write log
            self.__write_log(state)

        def on_train_end(self, args, state, control, logs=None, **kwargs):
            '''Write log after training'''
            self.__write_log(state)

    # Training arg
    training_args = hparams['pretraining_args']

    # Finetune
    trainer = Custom_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[
            hparams['early_stopping'],
            PrinterCallback()]
    )
    trainer.train()

    # Save ckpt
    if not os.path.exists(hparams['pretrain_ckpts']):
        os.makedirs(hparams['pretrain_ckpts'])
    trainer.save_model(hparams['pretrain_ckpts'])

    # Eval
    trainer.evaluate()


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Parse hparams
    hparams = parse_arguments(sys.argv[1:])

    # Set seed
    np.random.seed(hparams['seed'])
    torch.manual_seed(hparams['seed'])
    torch.cuda.manual_seed_all(hparams['seed'])

    # Set Env
    os.environ["CUDA_VISIBLE_DEVICES"]=hparams['GPUs']
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:256"

    # Create exp dir
    if not os.path.exists(hparams['pretrain_exp_dir']):
        os.makedirs(hparams['pretrain_exp_dir'])

    # Save yaml config for reference
    with open(f"{hparams['pretrain_exp_dir']}/log_exp-conf.yaml", 'w+') as fout:
        yaml.dump(hparams, fout,
            default_flow_style=False)

    # Run exp
    pretrain(hparams)
