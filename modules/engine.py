import torch

import pandas as pd
import torch.nn as nn

from typing import Iterable
from modules.loss import PHOSCLoss

from modules.utils import get_map_dict
from modules.datasets import CharacterCounterDataset


from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: PHOSCLoss,
                    dataloader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):

    model.train(True)

    n_batches = len(dataloader)
    batch = 1
    loss_over_epoch = 0

    pbar = tqdm(dataloader)

    for batch in pbar:
        # Putting images and targets on given device
        batch['image'] = batch['image'].to(device, non_blocking=True)
        batch['y_vectors']['phos'] = batch['y_vectors']['phos'].to(device, non_blocking=True)
        batch['y_vectors']['phoc'] = batch['y_vectors']['phoc'].to(device, non_blocking=True)
        batch['y_vectors']['phosc'] = batch['y_vectors']['phosc'].to(device, non_blocking=True)

        # zeroing gradients before next pass through
        model.zero_grad()

        # passing images in batch through model
        outputs = model(batch['image'])

        # calculating loss and backpropagation the loss through the network
        loss = criterion(outputs, batch['y_vectors'])
        loss.backward()

        # adjusting weight according to backpropagation
        optimizer.step()

        # accumulating loss over complete epoch
        loss_over_epoch += loss.item()

        pbar.set_description(f'loss: {loss}')

    # mean loss for the epoch
    mean_loss = loss_over_epoch / n_batches

    return mean_loss


# tensorflow accuracy function, modified for pytorch
@torch.no_grad()
def zslAccuracyTest(model, dataloader: Iterable, device: torch.device):
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

    # this will not work with the current dataloader
    for batch in tqdm(dataloader):

        batch['image'] = batch['image'].to(device, non_blocking=True)
        batch['y_vectors']['phos'] = batch['y_vectors']['phos'].to(device, non_blocking=True)
        batch['y_vectors']['phoc'] = batch['y_vectors']['phoc'].to(device, non_blocking=True)
        batch['y_vectors']['phosc'] = batch['y_vectors']['phosc'].to(device, non_blocking=True)

        vector_dict = model(batch['image'])
        vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1)

        phosc_size = vectors.shape[1]

        for i in range(len(batch['word'])):
            target_word = batch['word'][i]
            pred_vector = vectors[i].view(-1, phosc_size)
            mx = -1

            for w in word_map:
                temp = torch.cosine_similarity(pred_vector, torch.tensor(word_map[w]).to(device))
                if temp > mx:
                    mx = temp
                    pred_word = w

            Predictions.append((batch['image'][i], target_word, pred_word))

            if pred_word == target_word:
                n_correct += 1
                acc_by_len[len(target_word)] += 1

            word_count_by_len[len(target_word)] += 1

    for w in acc_by_len:
        if acc_by_len[w] != 0:
            acc_by_len[w] = acc_by_len[w] / word_count_by_len[w] * 100

    df = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])

    acc = n_correct / no_of_images

    print('acc:', acc)

    return acc, df, acc_by_len

# reserved for testing the character counter model
@torch.no_grad()
def test_accuracy(model: torch.nn.Module, dataloader: CharacterCounterDataset, device: torch.device):
    # set model in evaluation mode. turns of dropout layers and other layers which only are used for training. same
    # as .train(False)
    model.eval()

    # how many correct classified images
    cnt = 0

    for samples, targets, _ in tqdm(dataloader):
        # puts tensors onto devices
        samples = samples.to(device)
        targets = targets.to(device)

        # pass image and get output vector
        output = model(samples)

        # get argmax for output and target
        argmax_output = torch.argmax(output, dim=1)
        argmax_target = torch.argmax(targets, dim=1)

        for i in range(len(argmax_output)):
            if argmax_output[i] == argmax_target[i]:
                cnt += 1
                
                # print(argmax_output[i], argmax_target[i])
                # print(output[i], targets[i])
        
    # print(cnt)
    # print(len(dataloader.dataset))
    # print(cnt / len(dataloader.dataset))
    # number of correct predicted / total number of samples
    return cnt / len(dataloader.dataset)