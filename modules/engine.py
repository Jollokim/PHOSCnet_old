<<<<<<< HEAD
import torch

import pandas as pd
import torch.nn as nn

from typing import Iterable
from modules.loss import PHOSCLoss
from utils import get_map_dict


from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: PHOSCLoss,
                    dataloader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):

    model.train(True)

    n_batches = len(dataloader)
    batch = 1
    loss_over_epoch = 0

    pbar = tqdm(dataloader)

    for samples, targets, _ in pbar:
        # Putting images and targets on given device
        samples = samples.to(device)
        targets = targets.to(device)

        # zeroing gradients before next pass through
        model.zero_grad()

        # passing images in batch through model
        outputs = model(samples)

        # calculating loss and backpropagation the loss through the network
        loss = criterion(outputs, targets)
        loss.backward()

        # adjusting weight according to backpropagation
        optimizer.step()

        # print(f'loss: {loss.item()}, step progression: {batch}/{n_batches}, epoch: {epoch}')

        batch += 1

        # accumulating loss over complete epoch
        loss_over_epoch += loss.item()

        pbar.set_description(f'loss: {loss}')

    # mean loss for the epoch
    mean_loss = loss_over_epoch / n_batches

    return mean_loss


# tensorflow accuracy function, modified for pytorch
@torch.no_grad()
def accuracy_test(model, dataloader: Iterable, device: torch.device):
    # set in model in training mode
    model.eval()

    # gets the dataframe with all images, words and word vectors
    df = dataloader.dataset.df_all

    # gets the word map dictionary
    word_map = get_map_dict(list(set(df['Word'])))

    # number of correct predicted
    n_correct = 0
    no_of_images = len(df)

    # accuracy per word length
    acc_by_len = dict()

    # number of words per word length
    word_count_by_len = dict()

    # fills up the 2 described dictionaries over
    for w in df['Word'].tolist():
        acc_by_len[len(w)] = 0
        word_count_by_len[len(w)] = 0

    # Predictions list
    Predictions = []

    for samples, targets, words in tqdm(dataloader):
        samples = samples.to(device)

        vector_dict = model(samples)
        vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1)

        for i in range(len(words)):
            target_word = words[i]
            pred_vector = vectors[i].view(-1, 769)
            mx = -1

            for w in word_map:
                temp = torch.cosine_similarity(pred_vector, torch.tensor(word_map[w]).to(device))
                if temp > mx:
                    mx = temp
                    pred_word = w

            Predictions.append((samples[i], target_word, pred_word))

            if pred_word == target_word:
                n_correct += 1
                acc_by_len[len(target_word)] += 1

            word_count_by_len[len(target_word)] += 1

    for w in acc_by_len:
        if acc_by_len[w] != 0:
            acc_by_len[w] = acc_by_len[w] / word_count_by_len[w] * 100

    df = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])

    acc = n_correct / no_of_images

    return acc, df, acc_by_len

    # # op = None
    # for i in range(no_of_images):
    #     img_path = os.path.join(dataloader.dataset.root_dir, df.iloc[i, 0])
    #     x = io.imread(img_path)
    #
    #     word = df['Word'].iloc[i]
    #     word_count_by_len[len(word)] += 1
    #
    #     x = transform(x).view(1, 3, 50, 250).to(device)
    #
    #     y_pred = model(x).to(device)
    #
    #     # print(y_pred.shape)
    #     # print(y_pred)
    #     mx = -1
    #     for w in word_map:
    #         temp = torch.cosine_similarity(y_pred, torch.tensor(word_map[w]).
    #                                        to(device))
    #         if temp > mx:
    #             mx = temp
    #             op = w
    #     # print(word,op,mx)
    #     conf_matrix[idx_true[len(word)]][idx_pred[len(op)]] += 1
    #     Predictions.append((df['Image'].iloc[i], word, op))
    #     if op == word:
    #         cnt += 1
    #         acc_by_len[len(word)] += 1
    # for w in acc_by_len:
    #     if acc_by_len[w] != 0:
    #         acc_by_len[w] = acc_by_len[w] / word_count_by_len[w] * 100
    # df = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])
=======
import torch

import pandas as pd
import torch.nn as nn

from typing import Iterable
from modules.loss import PHOSCLoss
from utils import get_map_dict


def train_one_epoch(model: torch.nn.Module, criterion: PHOSCLoss,
                    dataloader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):

    model.train(True)

    n_batches = len(dataloader)
    batch = 1
    loss_over_epoch = 0
    for samples, targets, _ in dataloader:
        # Putting images and targets on given device
        samples = samples.to(device)
        targets = targets.to(device)

        # zeroing gradients before next pass through
        model.zero_grad()

        # passing images in batch through model
        outputs = model(samples)

        # calculating loss and backpropagation the loss through the network
        loss = criterion(outputs, targets)
        loss.backward()

        # adjusting weight according to backpropagation
        optimizer.step()

        print(f'loss: {loss.item()}, step progression: {batch}/{n_batches}, epoch: {epoch}')

        batch += 1

        # accumulating loss over complete epoch
        loss_over_epoch += loss.item()

    # mean loss for the epoch
    mean_loss = loss_over_epoch / n_batches

    return mean_loss


# tensorflow accuracy function, modified for pytorch
@torch.no_grad()
def accuracy_test(model, dataloader: Iterable, device: torch.device):
    # set in model in training mode
    model.eval()

    # gets the dataframe with all images, words and word vectors
    df = dataloader.dataset.df_all

    # gets the word map dictionary
    word_map = get_map_dict(list(set(df['Word'])))

    # number of correct predicted
    n_correct = 0
    no_of_images = len(df)

    # accuracy per word length
    acc_by_len = dict()

    # number of words per word length
    word_count_by_len = dict()

    # fills up the 2 described dictionaries over
    for w in df['Word'].tolist():
        acc_by_len[len(w)] = 0
        word_count_by_len[len(w)] = 0

    # Predictions list
    Predictions = []

    for samples, targets, words in dataloader:
        samples = samples.to(device)

        vector_dict = model(samples)
        vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1)

        for i in range(len(words)):
            target_word = words[i]
            pred_vector = vectors[i].view(-1, 769)
            mx = -1

            for w in word_map:
                temp = torch.cosine_similarity(pred_vector, torch.tensor(word_map[w]).to(device))
                if temp > mx:
                    mx = temp
                    pred_word = w

            Predictions.append((samples[i], target_word, pred_word))

            if pred_word == target_word:
                n_correct += 1
                acc_by_len[len(target_word)] += 1

            word_count_by_len[len(target_word)] += 1

    for w in acc_by_len:
        if acc_by_len[w] != 0:
            acc_by_len[w] = acc_by_len[w] / word_count_by_len[w] * 100

    df = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])

    acc = n_correct / no_of_images

    return acc, df, acc_by_len

    # # op = None
    # for i in range(no_of_images):
    #     img_path = os.path.join(dataloader.dataset.root_dir, df.iloc[i, 0])
    #     x = io.imread(img_path)
    #
    #     word = df['Word'].iloc[i]
    #     word_count_by_len[len(word)] += 1
    #
    #     x = transform(x).view(1, 3, 50, 250).to(device)
    #
    #     y_pred = model(x).to(device)
    #
    #     # print(y_pred.shape)
    #     # print(y_pred)
    #     mx = -1
    #     for w in word_map:
    #         temp = torch.cosine_similarity(y_pred, torch.tensor(word_map[w]).
    #                                        to(device))
    #         if temp > mx:
    #             mx = temp
    #             op = w
    #     # print(word,op,mx)
    #     conf_matrix[idx_true[len(word)]][idx_pred[len(op)]] += 1
    #     Predictions.append((df['Image'].iloc[i], word, op))
    #     if op == word:
    #         cnt += 1
    #         acc_by_len[len(word)] += 1
    # for w in acc_by_len:
    #     if acc_by_len[w] != 0:
    #         acc_by_len[w] = acc_by_len[w] / word_count_by_len[w] * 100
    # df = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])
>>>>>>> 1620f285b7edfbd38fd52ede974c863d87319f3e
