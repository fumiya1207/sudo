from PIL import Image
import os, glob
import numpy as np
#from sklearn import cross_validation
#from sklearn import model_selection
import cv2
import random
import time

classes = ["dog", "monkey"]
num_classes = len(classes)
image_size = (25,25)
num_limit_data = 88
num_testdata = 0

G1 = np.arange(256, dtype = 'uint8' )
G2 = np.arange(256, dtype = 'uint8' )
G3 = np.arange(256, dtype = 'uint8' )
G4 = np.arange(256, dtype = 'uint8' )

min_table = 50
max_table = 205
diff_table = max_table - min_table

LUT_HC = np.arange(256, dtype = 'uint8' )
LUT_LC = np.arange(256, dtype = 'uint8' )

# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    LUT_HC[i] = 255

# ローコントラストLUT作成
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255

gamma1 = 0.75
gamma2 = 1.0
gamma3 = 1.25
gamma4 = 1.5
for i in range(256):
    G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
    G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
    G3[i] = 255 * pow(float(i) / 255, 1.0 / gamma3)
    G4[i] = 255 * pow(float(i) / 255, 1.0 / gamma4)

#average_square1 = (4, 4)
average_square2 = (3, 3)
average_square3 = (2, 2)



def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def main():
    #画像の読み込み
    print("start")
    t1 = time.time()
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    for index, classlabel in enumerate(classes):

        photos_dir = "./test_data/" + classlabel
        files = sorted(glob.glob(photos_dir + "/*.jpg"))
        for i, file in enumerate(files):
            if i >= num_limit_data: break
            #print(file)
            image = Image.open(file)
            image = image.convert("RGB")
            #image = image.resize((image_size, image_size))


            if i < num_testdata:
                for j in range(9):

                    #回転
                    img = image.rotate(j*40)
                    #リサイズ1
                    img = np.asarray(img)
                    img = cv2.resize(img,(100,100))
                    #トリミング
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(np.uint8(img))
                    ran_width=random.randint(60,80)
                    ran_height=random.randint(60,80)
                    img=crop_center(img,ran_width,ran_height)
                    #リサイズ2
                    img = np.asarray(img)
                    img = cv2.resize(img,image_size)
                    X_train.append(img)
                    Y_train.append(index)
                    #変換


                    data = cv2.LUT(img, G1)
                    X_test.append(data)
                    Y_test.append(index)
                    data = cv2.LUT(img, G2)
                    X_test.append(data)
                    Y_test.append(index)
                    data = cv2.LUT(img, G3)
                    X_test.append(data)
                    Y_test.append(index)
                    data = cv2.LUT(img, G4)
                    X_test.append(data)
                    Y_test.append(index)

                    data = cv2.LUT(img, LUT_HC)
                    X_test.append(data)
                    Y_test.append(index)
                    data = cv2.LUT(img, LUT_LC)
                    X_test.append(data)
                    Y_test.append(index)


                    data = cv2.blur(img, average_square2)
                    X_test.append(data)
                    Y_test.append(index)
                    data = cv2.blur(img, average_square3)
                    X_test.append(data)
                    Y_test.append(index)

                    row, col, ch = img.shape
                    mean = 0
                    sigma = 15
                    gauss = np.random.normal(mean, sigma, (row, col, ch))
                    gauss = gauss.reshape(row, col, ch)
                    data = img + gauss
                    X_test.append(data)
                    Y_test.append(index)


                    row, col, ch = img.shape
                    s_vs_p = 0.5
                    amount = 0.004
                    sp_img = img.copy()
                    # 塩モード
                    num_salt = np.ceil(amount * img.size * s_vs_p)
                    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
                    sp_img[coords[:-1]] = (255, 255, 255)
                    # 胡椒モード
                    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
                    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
                    sp_img[coords[:-1]] = (0, 0, 0)
                    data = cv2.resize(sp_img,image_size)
                    X_test.append(data)
                    Y_test.append(index)


            else:
                for j in range(9):
                    #回転
                    img = image.rotate(j*40)
                    #リサイズ1
                    img = np.asarray(img)
                    img = cv2.resize(img,(100,100))
                    #トリミング
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(np.uint8(img))
                    ran_width=random.randint(60,80)
                    ran_height=random.randint(60,80)
                    img=crop_center(img,ran_width,ran_height)
                    #リサイズ2
                    img = np.asarray(img)
                    #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img,image_size)
                    #cv2.imwrite("./test/" + str(i) + classlabel + ".jpg", img)
                    X_train.append(img)
                    Y_train.append(index)
                    #変換



                    data = cv2.LUT(img, G1)
                    X_train.append(data)
                    Y_train.append(index)
                    data = cv2.LUT(img, G2)
                    X_train.append(data)
                    Y_train.append(index)
                    data = cv2.LUT(img, G3)
                    X_train.append(data)
                    Y_train.append(index)
                    data = cv2.LUT(img, G4)
                    X_train.append(data)
                    Y_train.append(index)

                    data = cv2.LUT(img, LUT_HC)
                    X_train.append(data)
                    Y_train.append(index)
                    data = cv2.LUT(img, LUT_LC)
                    X_train.append(data)
                    Y_train.append(index)


                    data = cv2.blur(img, average_square2)
                    X_train.append(data)
                    Y_train.append(index)
                    #cv2.imwrite("./test/" + str(i) + classlabel + "_blur.jpg", data)
                    data = cv2.blur(img, average_square3)
                    X_train.append(data)
                    Y_train.append(index)



                    row, col, ch = img.shape
                    mean = 0
                    sigma = 15
                    gauss = np.random.normal(mean, sigma, (row, col, ch))
                    gauss = gauss.reshape(row, col, ch)
                    data = img + gauss
                    #cv2.imwrite("./test/" + str(i) + classlabel + "_gauss.jpg", data)
                    X_train.append(data)
                    Y_train.append(index)



                    #row, col, ch = img.shape
                    s_vs_p = 0.5
                    amount = 0.004
                    sp_img = img.copy()
                    # 塩モード
                    num_salt = np.ceil(amount * img.size * s_vs_p)
                    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
                    sp_img[coords[:-1]] = (255, 255, 255)
                    # 胡椒モード
                    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
                    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
                    sp_img[coords[:-1]] = (0, 0, 0)
                    data = cv2.resize(sp_img,image_size)
                    #cv2.imwrite("./test/" + str(i) + classlabel + "_salt.jpg", data)
                    X_train.append(data)
                    Y_train.append(index)
                    cv2.imwrite("./test/" + str(i) + classlabel + "_blur.jpg", data)


    #X = np.array(X)
    #Y = np.array(Y)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(Y_train)
    y_test  = np.array(Y_test)

    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y)
    xy = (X_train, X_test, y_train, y_test)
    np.save("./data_dog_monkey.npy", xy)

    t2 = time.time()
    print("finish")
    elapsed_time = t2-t1
    print(f"経過時間：{elapsed_time}")


if __name__ == '__main__':
    main()

