import os
import random

import pandas as pd

import augmentation

import cv2 as cv

from tqdm import tqdm

import shutil


def count_words(df: pd.DataFrame) -> dict:
    word_counter = {}

    for word in df['Word']:
        if not word in word_counter.keys():
            word_counter[word] = 1
        else:
            word_counter[word] += 1

    return word_counter


def validate_dataset(df: pd.DataFrame, path: str):
    for img in tqdm(df['Image']):
        if not os.path.exists(os.path.join(path, img)):
            raise FileNotFoundError(os.path.join(path, img))


def copy_df_content_to_folder(df: pd.DataFrame, in_path: str, out_path: str):
    for file in tqdm(df['Image']):
        img = cv.imread(os.path.join(in_path, file))

        img = augmentation.resize_img(img)

        cv.imwrite(os.path.join(out_path, file), img)


def augment_word_class(df: pd.DataFrame,
                       word: str,
                       current_n: int,
                       in_folder_path: str,
                       out_folder_path: str,
                       total_n: int,
                       noise_variability:int=30,
                       max_shear_factor:int=2)->pd.DataFrame:

    iterations = total_n-current_n

    if iterations <= 0:
        return df

    df_sub = df[df['Word'] == word]

    for i in range(iterations):
        img_i = random.randint(0, len(df_sub)-1)

        img_fname = df_sub.iloc[img_i, 0]

        shearing_factor = random.random() * max_shear_factor

        img = cv.imread(f'{in_folder_path}/{img_fname}')

        img = augmentation.resize_img(img)
        img = augmentation.shear_image(img, shearing_factor)
        img = augmentation.resize_img(img)
        img = augmentation.noise_image(img, noise_variability)

        cv.imwrite(f'{out_folder_path}/{img_fname}_aug{i}.png', img)

        new_sample = pd.DataFrame([{'Image': f'{img_fname}_aug{i}.png', 'Word': word}])

        df = pd.concat([df, new_sample])

    return df



def main():
    random.seed(1)

    in_folder = r'image_data/IamSplit/trimmed_data/train'
    out_folder = r'image_data/IamSplit/augmented_data/train'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    df = pd.read_csv(f'{in_folder}.csv')
    word_dict = count_words(df)

    copy_df_content_to_folder(df, in_folder, out_folder)
    validate_dataset(df, out_folder)

    for word in tqdm(word_dict.keys()):
        df = augment_word_class(
            df, word, word_dict[word], in_folder, out_folder, 50)

    
    df.to_csv(f'{out_folder}.csv', index=False)

    validate_dataset(df, out_folder)


if __name__ == '__main__':
    main()
