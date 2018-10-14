#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:35:35 2018

@author: srinivas
"""

from imageio import imread
import numpy as np
import pandas as pd
import os
root = './RPS_data' # or ‘./test’ depending on for which the CSV is being created

# go through each directory in the root folder given above
for directory, subdirectories, files in os.walk(root):
# go through each file in that directory
    for file in files:
    # read the image file and extract its pixels
        print(file)
        im = imread(os.path.join(directory,file))
        value = im.flatten()
# I renamed the folders containing digits to the contained digit itself. For example, digit_0 folder was renamed to 0.
# so taking the 9th value of the folder gave the digit (i.e. "./train/8" ==> 9th value is 8), which was inserted into the first column of the dataset.
        value = np.hstack((directory[11:],value))
        df = pd.DataFrame(value).T
        df = df.sample(frac=1) # shuffle the dataset
        with open('train_foo.csv', 'a') as dataset:
            df.to_csv(dataset, header=False, index=False)
            
#building the model 
import numpy as np
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import pandas as pd

def keras_model(image_x, image_y):
    num_of_classes = 4
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "train_RPS.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    callbacks_list.append(TensorBoard(log_dir='RPS_logs'))

    return model, callbacks_list


def loadData():
    data = pd.read_csv("train_foo_RPS.csv")
    dataset = np.array(data)
    np.random.shuffle(dataset)
    features = dataset[:, 1:2501]
    features = features / 255.
    labels = dataset[:, 0]
    labels = labels.reshape(labels.shape[0], 1)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.2)
    return train_x, test_x, train_y, test_y


def reshapeData(train_x, test_x, train_y, test_y):
    train_y = np_utils.to_categorical(train_y)
    test_y = np_utils.to_categorical(test_y)
    train_x = train_x.reshape(train_x.shape[0], 50, 50, 1)
    test_x = test_x.reshape(test_x.shape[0], 50, 50, 1)
    return train_x, test_x, train_y, test_y


def printInfo(train_x, test_x, train_y, test_y):
    print("number of training examples = " + str(train_x.shape[0]))
    print("number of test examples = " + str(test_x.shape[0]))
    print("X_train shape: " + str(train_x.shape))
    print("Y_train shape: " + str(train_y.shape))
    print("X_test shape: " + str(test_x.shape))
    print("Y_test shape: " + str(test_y.shape))


def main():
    train_x, test_x, train_y, test_y = loadData()
    printInfo(train_x, test_x, train_y, test_y)
    print(len(train_x[0]))
    train_x, test_x, train_y, test_y = reshapeData(train_x, test_x, train_y, test_y)
    printInfo(train_x, test_x, train_y, test_y)
    model, callbacks_list = keras_model(train_x.shape[1], train_x.shape[2])
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=2, batch_size=64,
              callbacks=callbacks_list)
    scores = model.evaluate(test_x, test_y, verbose=1)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    print_summary(model)

    model.save('RPS.h5')


if __name__ == '__main__':
    main()


import cv2
from keras.models import load_model
import numpy as np
import os
from random import randint

model = load_model('RPS.h5')


def calcResult(pred_class, cpu):
    if pred_class == cpu:
        return 'draw'
    if pred_class == 1 and (cpu == 3):
        return 'user'
    if pred_class == 2 and (cpu == 1 ):
        return 'user'
    if pred_class == 3 and (cpu == 2):
        return 'user'
    return 'cpu'


def main():
    flag = 0
    result = ''
    emojis = get_emojis()
    cap = cv2.VideoCapture(0)
    x, y, w, h = 300, 50, 350, 350

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))
        res = cv2.bitwise_and(img, img, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=2)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)

        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                if flag == 0:
                    cpu = (randint(1,3))
                    flag = 1
                x, y, w1, h1 = cv2.boundingRect(contour)
                newImage = thresh[y:y + h1, x:x + w1]
                newImage = cv2.resize(newImage, (50, 50))
                pred_probab, pred_class = keras_predict(model, newImage)
                print(pred_class, pred_probab)
                img = overlay(img, emojis[pred_class], 370, 50, 90, 90)
                img = overlay(img, emojis[cpu], 530, 50, 90, 90)
                result = calcResult(pred_class, cpu)

        elif len(contours) == 0:
            flag = 0
        x, y, w, h = 300, 50, 350, 350
        cv2.putText(img, 'USER', (380, 40),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, 'CPU', (550, 40),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, 'Result : ', (420, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if result == 'user':
            cv2.putText(img, 'USER', (530, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif result=='cpu':
            cv2.putText(img, 'CPU', (530, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif result=='draw':
            cv2.putText(img, 'DRAW', (530, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            pass
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break


def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def get_emojis():
    emojis_folder = 'RPS_emo/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder + str(emoji) + '.png', -1))
    return emojis


def overlay(image, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y + h, x:x + w] = blend_transparent(image[y:y + h, x:x + w], emoji)
    except:
        pass
    return image


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
main()


            
