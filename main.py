import argparse
import torch
import os

from modules import models, residualmodels

from timm import create_model
from torchsummary import summary
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.datasets import phosc_dataset
from modules.engine import train_one_epoch, zslAccuracyTest
from modules.loss import PHOSCLoss

import torch.nn as nn

# function for defining all the commandline parameters
def get_args_parser():
    parser = argparse.ArgumentParser('Main', add_help=False)

    # Model mode:
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'pass'], required=True,
                        help='train or test a model')
    # Model settings
    parser.add_argument('--name', type=str, help='Name of run')
    parser.add_argument('--model', type=str, help='Name of model')
    parser.add_argument('--pretrained_weights', type=str, help='the path to pretrained weights file')

    # Dataset folder paths
    parser.add_argument('--train_csv', type=str, help='The train csv')
    parser.add_argument('--train_folder', type=str, help='The train root folder')
    parser.add_argument('--valid_csv', type=str, help='The valid csv')
    parser.add_argument('--valid_folder', type=str, help='The valid root folder')

    parser.add_argument('--test_csv_seen', type=str, help='The seen test csv')
    parser.add_argument('--test_folder_seen', type=str, help='The seen test root folder')

    parser.add_argument('--test_csv_unseen', type=str, help='The unseen test csv')
    parser.add_argument('--test_folder_unseen', type=str, help='The unseen test root folder')

    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples per iteration in the epoch')
    parser.add_argument('--num_workers', default=5, type=int)

    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')

    # trainng related parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')

    # model related
    parser.add_argument('--phos_size', type=int, default=165, help='Phos representation size')
    parser.add_argument('--phoc_size', type=int, default=604, help='Phoc representation size')
    parser.add_argument('--language', type=str, default='eng', choices=['eng', 'nor'], help='language which help decide which phosc representation to use')


    return parser


def main(args):
    print('Creating dataset...')
    if args.mode == 'train':
        dataset_train = phosc_dataset(args.train_csv,
                                      args.train_folder,
                                      args.language,
                                      transforms.ToTensor())

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True
        )

        validate_model = False

        if args.valid_csv is not None or args.valid_folder is not None:
            validate_model = True

            dataset_valid = phosc_dataset(args.valid_csv,
                                          args.valid_folder,
                                          args.language,
                                          transforms.ToTensor())

            data_loader_valid = torch.utils.data.DataLoader(
                dataset_valid,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                drop_last=False,
                shuffle=True
            )

    elif args.mode == 'test':
        dataset_test_seen = phosc_dataset(args.test_csv_seen,
                                     args.test_folder_seen,
                                     args.language,
                                     transforms.ToTensor())

        data_loader_test_seen = torch.utils.data.DataLoader(
            dataset_test_seen,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True
        )

        dataset_test_unseen = phosc_dataset(args.test_csv_unseen,
                                     args.test_folder_unseen,
                                     args.language,
                                     transforms.ToTensor())

        data_loader_test_unseen = torch.utils.data.DataLoader(
            dataset_test_unseen,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True
        )

    # setting the device to do stuff on
    print('Training on GPU:', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: add phos and phoc size to create_model
    model = create_model(args.model, phos_size=args.phos_size, phoc_size=args.phoc_size).to(device)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # print summary of model
    summary(model, (3, 50, 250))

    def training():
        if not os.path.exists(f'{args.name}/'):
            os.mkdir(args.name)

        with open(args.name + '/' + 'log.csv', 'a') as f:
            f.write('epoch,loss,acc,lr\n')

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)

        scheduler = ReduceLROnPlateau(opt, 'max', factor=0.25, patience=5, verbose=True, threshold=0.0001, cooldown=2,
                                      min_lr=1e-12)

        criterion = PHOSCLoss()

        mx_acc = 0
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            mean_loss = train_one_epoch(model, criterion, data_loader_train, opt, device, epoch)

            acc = -1
            if validate_model:
                acc, _, __ = zslAccuracyTest(model, data_loader_valid, device)

                if acc > mx_acc:
                    mx_acc = acc
                    # removes previous best weights
                    if os.path.exists(f'{args.name}/epoch{best_epoch}.pt'):
                        os.remove(f'{args.name}/epoch{best_epoch}.pt')

                    best_epoch = epoch

                    torch.save(model.state_dict(), f'{args.name}/epoch{best_epoch}.pt')

                scheduler.step(acc)

            with open(args.name + '/' + 'log.csv', 'a') as f:
                f.write(f'{epoch},{mean_loss},{acc},{opt.param_groups[0]["lr"]}\n')

            

        torch.save(model.state_dict(), f'{args.name}/epochs.pt')

    def testing():
        model.load_state_dict(torch.load(args.pretrained_weights))

        acc_seen, _, __ = zslAccuracyTest(model, data_loader_test_seen, device)
        acc_unseen, _, __ = zslAccuracyTest(model, data_loader_test_unseen, device)

        with open(args.name + '/' + 'testresults.txt', 'a') as f:
            f.write(f'{args.model} test results\n')
            f.write(f'Seen acc: {acc_seen}\n')
            f.write(f'Unseen acc: {acc_unseen}\n')

        print(f'accuracies of model: {args.model}')
        print('Seen accuracies:', acc_seen)
        print('Unseen accuracies:', acc_unseen)

    if args.mode == 'train':
        training()
    elif args.mode == 'test':
        testing()


# program start
if __name__ == '__main__':
    # creates commandline parser
    arg_parser = argparse.ArgumentParser('train ', parents=[get_args_parser()])
    args = arg_parser.parse_args()

    # passes the commandline argument to the main function
    main(args)
