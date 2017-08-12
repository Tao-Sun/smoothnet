import os
import argparse
import glob
import cv2
import numpy as np


def resize_img(img, scale):
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    resized_img = cv2.resize(blurred_img, (0, 0), fx=0.5, fy=0.5)
        
    if scale == 0.25:
        blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
        resized_img = cv2.resize(blurred_img, (0, 0), fx=0.5, fy=0.5)

    normalized_img = cv2.normalize(resized_img, dst=resized_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    result_img = np.uint8(normalized_img * 255)
    
    return result_img


def resize_images(input_dir, output_dir, scale=0.5, apply_clahe=False):
    for _, _, files in os.walk(input_dir):
        for f in files:
            if f.index('.png') > 0:
                img_file = f
                print(img_file)
                img = cv2.imread(input_dir + '/' + img_file)
                #print (img.shape)

                result_img = resize_img(img, scale)

                if apply_clahe:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                    result_img = clahe.apply(result_img)

                cv2.imwrite(output_dir + '/' + img_file, result_img)


def resize_labels(input_dir, output_dir, class_count, scale=0.5, apply_clahe=False):
    for _, _, files in os.walk(input_dir):
        for f in files:
            if f.index('.png') > 0:
                label_file = f
                print(label_file)
                label_img = cv2.imread(input_dir + '/' + label_file, 0)
                label_img_height = label_img.shape[0]
                label_img_width = label_img.shape[1]
                label_pixel_list = np.reshape(label_img, (label_img_height * label_img_width))

                resized_img_height = int(round(label_img_height * scale))
                resized_img_width = int(round(label_img_width * scale))
                stacked_class_img = np.array([]).reshape((0, resized_img_height, resized_img_width))

                for class_value in range(class_count):
                    #print('class value:' + str(class_value))
                    class_label_list = [(0.0 if label_pixel_list[i] != class_value else 255.0) for i, label in enumerate(label_pixel_list)]
                    class_label_img = np.reshape(class_label_list, (label_img.shape[0], label_img.shape[1]))
                    #print(class_label_img)

                    resized_class_img = resize_img(class_label_img, scale)

                    if apply_clahe:
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                        resized_class_img = clahe.apply(resized_class_img)

                    resized_class_img = np.reshape(resized_class_img, (1, resized_class_img.shape[0], resized_class_img.shape[1]))
                    stacked_class_img = np.vstack((stacked_class_img, resized_class_img))

                result_img = np.argmax(stacked_class_img, axis=0)
                cv2.imwrite(output_dir + '/' + label_file, result_img)


FLAGS = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        default='images',
        help="Types: 'images' or 'labels'."
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='Scale of the resizing.'
    )
    parser.add_argument(
        '--class_count',
        type=int,
        default=32,
        help='Total class count.'
    )
    parser.add_argument(
        '--improve_contrast',
        type=bool,
        default=False,
        help='Whether to improve contrast.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/data',
        help='Directory of the data.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/tmp/data',
        help='Output dir.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    
    input_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir
    scale = FLAGS.scale
    improve_contrast = FLAGS.improve_contrast

    if FLAGS.type == 'images':
        print("Starting resizing images...")
        resize_images(input_dir, output_dir, scale, improve_contrast)
    else:
        resize_labels(input_dir, output_dir, FLAGS.class_count, scale, improve_contrast)
