import os
import cv2 as cv
from tqdm import tqdm


def main():
    splits = [
        r'image_data/IamSplit/augmented_data/valid',
        r'image_data/IamSplit/augmented_data/test'
    ]

    for split in splits:
        in_folder_path = split
        output_folder_path = split

        for img_name in tqdm(os.listdir(in_folder_path)):
            img = cv.imread(os.path.join(in_folder_path, img_name))

            img = cv.resize(img, (250, 50))

            cv.imwrite(os.path.join(output_folder_path, img_name), img)


if __name__ == '__main__':
    main()