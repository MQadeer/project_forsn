#!/usr/bin/env python3

import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import os
import argparse
import numpy as np
import cv2
from skimage.util import random_noise

def apply_image_filters(bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    rand_bright_ship = np.random.randint(-10, 10)
    h, s, v = cv2.split(hsv)
    lim = 255 - rand_bright_ship
    v[v > lim] = 255
    np.add(v[v <= lim], rand_bright_ship, out=v[v <= lim], casting="unsafe")
    final_hsv = cv2.merge((h, s, v))
    bgr_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    noise_img = random_noise(bgr_image, mode='gaussian')
    bgr_image = (255 * noise_img).astype(np.uint8)
    return bgr_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input-data-dir', type=str, required=True,
                        help='Path to a directory with face images (with dedicated subdirectories for each person)')
    parser.add_argument('-od', '--output-data-dir', type=str, required=True,
                        help='Path to a directory where the augmented data will be saved (with dedicated subdirectories for each person)')
    parser.add_argument('-i', '--images-per-person-image', type=int, required=True,
                        help='Number of augmented images to generate for each image of a person')
    args = parser.parse_args()

    person_dir = args.input_data_dir
    augmented_person_dir = args.output_data_dir
    augmented_images_per_image = args.images_per_person_image
    for subject in os.listdir(person_dir):
        subject_dir_path = os.path.join(person_dir, subject)
        augmented_subject_dir_path = os.path.join(augmented_person_dir, subject)
        os.mkdir(augmented_subject_dir_path)

        print('Processing {0}'.format(subject))
        for f in os.listdir(subject_dir_path):
            image_path = os.path.join(subject_dir_path, f)
            img_id = f.split('.')[0]
            for i in range(augmented_images_per_image):
                augmented_image_path = os.path.join(augmented_subject_dir_path,
                                                    img_id + '_{0}.jpg'.format(i+1))

                bgr_image = cv2.imread(image_path)
                augmented_image = apply_image_filters(bgr_image)
                cv2.imwrite(augmented_image_path, augmented_image)
