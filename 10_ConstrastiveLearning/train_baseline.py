import sys
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn

from hyperpyyaml import load_hyperpyyaml


from src.datasets import Downstream_Dset, unpickle, lab_dict
from src.models import DownstreamModel
from src.trainers import DownstreamTrainer
from src.utils import compute_metrics


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

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Parse hparams
    hparams = parse_arguments(sys.argv[1:])

    # Set seed
    np.random.seed(hparams['seed'])
    torch.manual_seed(hparams['seed'])
    device='cuda:0' if torch.cuda.is_available() else 'cpu'

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

    # Train
    def train_downstream():
        # Load datasets
        train_dset = Downstream_Dset(train_images, train_labels,
            mode='train',
            mean=MEAN, std=STD)
        val_dset = Downstream_Dset(val_images, val_labels,
            mode='valid',
            mean=MEAN, std=STD)

        # Load pretrained
        model = DownstreamModel(
            premodel='resnet50',
            num_classes=len(lab_dict)).to(device)

        # Optmizer
        optimizer = torch.optim.SGD(
            [params for params in model.parameters() if params.requires_grad],
            lr=hparams['downstream_lr'],
            momentum=0.9)

        # schedulers
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch : (epoch+1)/10.0,
            verbose=True)
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1, gamma=0.98, last_epoch=-1, verbose=True)

        # LOSS FUNCTION
        criterion = nn.CrossEntropyLoss()

        # Train
        trainer = DownstreamTrainer(
            model=model, optimizer=optimizer, criterion=criterion,
            main_scheduler=main_scheduler, warmup_scheduler=warmup_scheduler,
            train_dataset=train_dset, val_dataset=val_dset, labels_dict=lab_dict,
            batch_size=hparams['batch_size'], num_workers=hparams['num_workers'],
            exp_name=hparams['exp_name'], device=device, mode='downstream')

        trainer.train(
            num_epochs=hparams['epochs'],
            eval_every_epochs=hparams['eval_every_epochs'],
            resume_from_chkpt=hparams['downstream_resume'],
            plot_feats=False)
    train_downstream()

    # Test
    def test_downstream():
        # Load test data
        test_images = np.array([], dtype=np.uint8).reshape((0,3072))
        test_labels = np.array([])

        data_dict = unpickle('DATA/test_batch')
        test_images = np.append(test_images, data_dict[b'data'],axis=0)
        test_labels = np.append(test_labels,  data_dict[b'labels'])

        test_images = test_images.reshape((-1,3,32,32)).astype(np.float)

        # Load datasets
        test_dataset = Downstream_Dset(test_images, test_labels,
            mode='test',
            mean=MEAN, std=STD)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=hparams['batch_size'], num_workers=hparams['num_workers'],
            drop_last=False, shuffle=False)

        # Load model checkpoint
        model = DownstreamModel(
            premodel='resnet50',
            num_classes=len(lab_dict)).to(device)
        model.load_state_dict(torch.load(
            f"exp/{hparams['exp_name']}/chkpts/best_downstreamed.pt")['model'])

        # Inference
        preds_list, labels_list = [], []
        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_dataloader, desc='Testing')):
                # Load batch
                imgs = batch['img'].to(device, dtype=torch.float)
                labels = batch['label'].to(device, dtype=torch.long)
                labels_list.append(labels.cpu().numpy())

                # positive pair, with encoding
                preds = model(imgs)
                preds_list.append(preds.cpu().numpy())

        # Compute metrics
        stats = compute_metrics(
            preds=np.concatenate(preds_list, axis=0),
            labels=np.concatenate(labels_list, axis=0))
        with open(f"exp/{hparams['exp_name']}/test_stats.txt", 'w+') as fout:
            fout.write(f"Test accuracy: {stats['accuracy']:.4f}\n")
            fout.write(f"Test precision: {stats['precision']:.4f}\n")
            fout.write(f"Test recall: {stats['recall']:.4f}\n")
            fout.write(f"Test f1-score: {stats['f1']:.4f}\n")
    test_downstream()
