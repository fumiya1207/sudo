from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras.utils import np_utils
import keras
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import os, glob
import cv2

%matplotlib inline
classes = ["dog", "monkey"]
num_classes = len(classes)
image_size = 25

"""
global j
j = 0
"""
#メインの関数を定義する
def main():
    t1 = time.time()
    print("start")

    X_train, X_test, y_train, y_test = np.load("./data_dog_monkey.npy")
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train)

    """
    model_eval(model, X_test, y_test)
    """
    t2 = time.time()
    print("finish")
    elapsed_time = t2-t1
    print(f"経過時間：{elapsed_time}")


def model_train(X, y):

    x_val = []
    y_val = []
    for index, classlabel in enumerate(classes):

        photos_dir = "./test_data/" + classlabel
        files = glob.glob(photos_dir + "/*.jpg")

        for i, file in enumerate(files):

                image = Image.open(file)
                image = image.convert("RGB")
                image = image.resize((image_size,image_size))
                data = np.asarray(image)#pythonの配列
                data = cv2.cvtColor(data,cv2.COLOR_RGB2BGR)
                x_val.append(data)
                y_val.append(index)
    x_val = np.array(x_val)#numpyの配列
    y_val = np.array(y_val)
    x_val = x_val.astype("float32") / 255
    y_val = np_utils.to_categorical(y_val, num_classes)


    model = Sequential()
    model.add(Conv2D(32,(3,3),padding = "same",input_shape = (25,25,3)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Activation("relu"))

    model.add(Dropout(0.30))

    """
    model.add(Conv2D(32,(3,3)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Activation("relu"))
    """

    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.30))

    """
    model.add(Conv2D(64,(3,3), padding = "same"))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Activation("relu"))

    model.add(Dropout(0.20))
    """
    model.add(Conv2D(64,(3,3)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Activation("relu"))

    model.add(Dropout(0.30))

    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.30))


    model.add(Flatten())


    model.add(Dense(1000))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    """
    model.add(Dense(300))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    """

    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes))

    model.add(Activation("sigmoid"))

    #model.load_weights('./animal_cnn_aug.h5')

    opt = keras.optimizers.rmsprop(lr = 0.0001, decay = 1e-6)

    model.summary()

    #model.load_weights('./animal_cnn_aug.h5')

    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])




    history = model.fit(X, y, batch_size = 64, epochs = 50,validation_data = (x_val,y_val))

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1,len(loss)+1)

    plt.plot(epochs,loss,"b",label="Training loss")
    plt.plot(epochs,val_loss,"g",label="Validation loss")
    plt.title("Training and Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    

    acc = history.history["acc"]
    val_acc = history.history["val_acc"]

    epochs = range(1,len(acc)+1)

    plt.plot(epochs,acc,"b",label="Training acc")
    plt.plot(epochs,val_acc,"g",label="Validation acc")
    plt.title("Training and Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.legend()
    plt.show()

    model.save_weights("./cnn_dog_monkey_weights2.h5")
    model.save("./cnn_dog_monkey2.h5")

    return model

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose = 1)
    print("Test loss: ", scores[0])
    print("Test Accuracy: ",scores[1])

if __name__ == "__main__":
    main()

