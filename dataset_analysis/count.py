import os
import pandas as pd

def find_max_length_word(df: pd.DataFrame)->int:
    mx_len = 0
    mx_word = ''
    for word in df['Word']:
        if mx_len < len(word):
            mx_len = len(word)
            mx_word = word
    return mx_len, mx_word

def main():
    df_train = pd.read_csv('image_data/IAM_Data/IAM_train.csv')
    df_valid = pd.read_csv('image_data/IAM_Data/IAM_valid.csv')
    df_test_seen = pd.read_csv('image_data/IAM_Data/IAM_test_seen.csv')
    df_test_unseen = pd.read_csv('image_data/IAM_Data/IAM_test_unseen.csv')

    print(df_train['Word'])

    print(find_max_length_word(df_train))
    print(find_max_length_word(df_valid))
    print(find_max_length_word(df_test_seen))
    print(find_max_length_word(df_test_unseen))

if __name__ == '__main__':
    main()