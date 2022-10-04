import optuna

from timm import create_model

import modules.models
from modules.datasets import phosc_dataset
import torch

import torch.nn as nn

from torchvision.transforms import transforms

from modules.engine import train_one_epoch, accuracy_test

from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.loss import PHOSCLoss

def objective(trial: optuna.trial.Trial):

    params = {
              'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
              'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1)
              }

    dataset_train = phosc_dataset(
                                    'image_data/IAM_Data/IAM_train.csv',
                                    'image_data/IAM_Data/IAM_train',
                                    transforms.ToTensor()
                                )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        num_workers=10,
        drop_last=False,
        shuffle=True
    )

    dataset_valid = phosc_dataset(
                                    'image_data/IAM_Data/IAM_valid_seen.csv',
                                    'image_data/IAM_Data/IAM_valid',
                                    transforms.ToTensor()
                                )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=64,
        num_workers=10,
        drop_last=False,
        shuffle=True
    )

    dataset_test_seen = phosc_dataset(
                                        'image_data/IAM_Data/IAM_test_seen.csv',
                                        'image_data/IAM_Data/IAM_test',
                                        transforms.ToTensor())

    data_loader_test_seen = torch.utils.data.DataLoader(
        dataset_test_seen,
        batch_size=64,
        num_workers=10,
        drop_last=False,
        shuffle=True
    )
    
    model = create_model('PHOSCnet_temporalpooling')

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    scheduler = ReduceLROnPlateau(opt, 'max', factor=0.25, patience=5, verbose=True, threshold=0.0001, cooldown=2,
                                    min_lr=1e-12)

    criterion = PHOSCLoss()

    mx_acc = 0
    best_epoch = 0
    for epoch in range(1, 30):
        mean_loss = train_one_epoch(model, criterion, data_loader_train, opt, device, epoch)

        acc = -1
        acc, _, __ = accuracy_test(model, data_loader_valid, device)

        if acc > mx_acc:
            mx_acc = acc

            best_epoch = epoch

            scheduler.step(acc)


    accuracy = accuracy_test(model, data_loader_test_seen)

    return accuracy


def main():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=15)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))


if __name__ == '__main__':
    main()