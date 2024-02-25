# -*- coding: utf-8 -*-
'''
https://github.com/V-Sher/Audio-Classification-HF/blob/main/scripts/audio_train.py
'''

import sys, os
import yaml
import json
from glob import glob
from hyperpyyaml import load_hyperpyyaml

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import \
    accuracy_score, recall_score, precision_score, f1_score

from src.models import \
    Resnet_SCL_Config, Resnet_SCL
from src.datasets import unpickle, lab_dict, Downstream_Dset

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

def train(hparams):
    train_images, train_labels, val_images, val_labels, MEAN, STD = load_training_data()

    # Load Datasets
    train_dset = Downstream_Dset(train_images, train_labels,
            mode='train',
            mean=MEAN, std=STD)
    eval_dset = Downstream_Dset(val_images, val_labels,
            mode='valid',
            mean=MEAN, std=STD)

    # Load model
    config = Resnet_SCL_Config(
        num_classes=len(lab_dict),
        mode='downstream')
    model = Resnet_SCL.from_pretrained(
        hparams['pretrain_ckpts'],
        config=config)

    # optimizer, scheduler, loss function
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=hparams['learning_rate'],
        weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer)

    class Custom_Trainer(Trainer): pass
        # def compute_loss(self, model, inputs, return_outputs=False):
        #     labels = inputs.pop("labels")
        #     outputs = model(**inputs)

        #     loss = criterion(outputs.logits, labels)
        #     return (loss, outputs) if return_outputs else loss

    # Eval metrics
    def compute_metrics(eval_pred, average='macro'):
        assert average in [None, 'micro', 'macro', 'weighted']

        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids

        # accuracy, precision, recall, f1-score
        acc = accuracy_score(
            y_true=labels, y_pred=predictions,
            normalize=True)
        r = recall_score(
            y_true=labels, y_pred=predictions,
            average=average, zero_division=0)
        p = precision_score(
            y_true=labels, y_pred=predictions,
            average=average, zero_division=0)
        f1 = f1_score(
            y_true=labels, y_pred=predictions,
            average=average, zero_division=0)

        return {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1 }

    class PrinterCallback(TrainerCallback):
        def __write_log(self, state):
            train_log, eval_log = [], []

            for e in state.log_history:
                e_keys = set(e)
                if "loss" in e_keys: train_log.append(e)
                elif "eval_loss" in e_keys: eval_log.append(e)
                elif "train_runtime" in e_keys:
                    with open(f"{hparams['downstream_exp_dir']}/trainer_info.json", 'w+', encoding='utf-8') as fin:
                        json.dump(e, fin, ensure_ascii=False, indent=4)

            if train_log != []:
                train_log_df = pd.DataFrame.from_dict(train_log) \
                    .sort_values("step", ascending=True) \
                    .reset_index(drop=True)
                train_log_df.to_csv(f"{hparams['downstream_exp_dir']}/log_trainset.csv", index=False)

            if eval_log != []:
                eval_log_df = pd.DataFrame.from_dict(eval_log) \
                    .sort_values("step", ascending=True) \
                    .reset_index(drop=True)
                eval_log_df.to_csv(f"{hparams['downstream_exp_dir']}/log_evalset.csv", index=False)

        def on_evaluate(self, args, state, control, logs=None, **kwargs):
            '''Write log after every eval round'''
            self.__write_log(state)
        def on_train_end(self, args, state, control, logs=None, **kwargs):
            '''Write log after training'''
            self.__write_log(state)

    # Training arg
    training_args = hparams['training_args']

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
    if not os.path.exists(hparams['test_ckpts']):
        os.makedirs(hparams['test_ckpts'])
    trainer.save_model(hparams['test_ckpts'])

    # Eval
    trainer.evaluate()

def load_testing_data():
    # Load test data
    test_images = np.array([], dtype=np.uint8).reshape((0,3072))
    test_labels = np.array([])

    data_dict = unpickle('DATA/test_batch')
    test_images = np.append(test_images, data_dict[b'data'],axis=0)
    test_labels = np.append(test_labels,  data_dict[b'labels'])

    test_images = test_images.reshape((-1,3,32,32)).astype(np.float)
    return test_images, test_labels

def test(hparams):
    _, _, _, _, MEAN, STD = load_training_data()
    test_images, test_labels = load_testing_data()

    # Create dset
    test_dataset = Downstream_Dset(test_images, test_labels,
        mode='test',
        mean=MEAN, std=STD)

    # Load model
    config = Resnet_SCL_Config(
        num_classes=len(lab_dict),
        mode='downstream')
    model = Resnet_SCL.from_pretrained(
        hparams['test_ckpts'], config=config)

    # Infer
    test_trainer = Trainer(
        model=model,
        args=hparams['testing_args'])
    preds, _ , _ = test_trainer.predict(test_dataset)

    # Predict
    y_preds = np.argmax(preds, axis=1).astype(int)

    # scores
    scores = pd.DataFrame(
        data=preds,
        columns=[ f"Class_{c}_score" for c in (np.arange(len(lab_dict))) ])

    # Prediction
    y_preds = pd.DataFrame(
        data=y_preds,
        columns=[ 'Prediction' ])
    test_labels = pd.DataFrame(
        data=test_labels,
        columns=[ 'label' ])

    # out
    test_df = pd.concat(
        [test_labels, scores, y_preds], axis=1)
    test_df.to_csv(f"{hparams['downstream_exp_dir']}/test_scores.csv", index=False)


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
    if not os.path.exists(hparams['downstream_exp_dir']):
        os.makedirs(hparams['downstream_exp_dir'])

    # Save yaml config for reference
    with open(f"{hparams['downstream_exp_dir']}/log_exp-conf.yaml", 'w+') as fout:
        yaml.dump(hparams, fout,
            default_flow_style=False)

    # Run exp
    train(hparams)
    test(hparams)
