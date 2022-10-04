import os
import random
import numpy as np
import cv2 as cv

# modified version of: https://github.com/anuj-rai-23/PHOSC-Zero-Shot-Word-Recognition/blob/main/aug_images.py
def noise_image(img, variability):
    deviation = variability*random.random()

    noise = np.int32(np.random.normal(0, deviation, img.shape))

    img += noise
    img = np.uint8(np.clip(img, 0., 255.))
    return img

def shear_image(img, x_shearing_factor):
    if x_shearing_factor >= 0:
        off_bound_x = (img.shape[1])+(x_shearing_factor*img.shape[0])
    else:
        raise NotImplementedError('shear_image has not been implemented for shearing facotr of -1 yet.')
    
    img_shear = np.full((int(img.shape[0]), int(off_bound_x), 3), 255, dtype=np.float64)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x = col + (x_shearing_factor*row)
            y = row
            

            # if x <= -1 or y <= -1:
            #     continue
            try:
                img_shear[int(y)][int(x)] = img[row, col]
            except IndexError:
                pass

    return img_shear

def resize_img(img):
    img = cv.resize(img, (250, 50)).copy()

    return img
            

def main():
    # print('train, test, validate, total')
    # print(len(os.listdir(r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#')), end=', ')
    # print(len(os.listdir(r'image_data/IamSplit/data/iamOffValCrops#07-2022-07-7#')), end=', ')
    # print(len(os.listdir(r'image_data/IamSplit/data/iamOffTestCrops#07-2022-07-7#')), end=', ')
    # print(len(os.listdir(r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#')) + 
    #     len(os.listdir(r'image_data/IamSplit/data/iamOffValCrops#07-2022-07-7#')) + 
    #     len(os.listdir(r'image_data/IamSplit/data/iamOffTestCrops#07-2022-07-7#')))

    # img_path = r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#/a01-000u.png_16454_[1405, 1140, 1469, 1175]_a_#07-2022-07-7#.png'
    img_path = r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#/a01-000u.png_16443_[395, 932, 836, 1032]_nominating_#07-2022-07-7#.png'

    img = cv.imread(img_path)

    cv.imwrite(r'image_data/test.png', img)

    img = cv.resize(img, (250, 50))

    img_sheared = shear_image(img, 2)

    img_sheared = cv.resize(img_sheared, (250, 50))

    cv.imwrite(r'image_data/sheared_test.png', img_sheared)

    img_noise = noise_image(img_sheared, 30)

    cv.imwrite(r'image_data/noise_test.png', img_noise)


if __name__ == "__main__":
    main()