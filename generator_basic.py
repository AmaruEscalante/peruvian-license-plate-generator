import os
import random
import cv2
import argparse
import numpy as np


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA): #mantain aspect ratio
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def random_bright(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("contorno.png")
        self.flag = cv2.imread("bandera.jpg")
        self.country = cv2.imread("peru.jpg")
        self.hyphen = cv2.imread("hyphen.jpg")

        # loading Number
        file_path = "./numbers/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.number_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number.append(img)
            self.number_list.append(file[:-4])  # [:-4] all elements except the last 4 -> .jpg

        # loading Char
        file_path = "./characters/"
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.Char1 = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1.append(img)
            self.char_list.append(file[:-4])  # [:-4] all elements except the last 4 -> .jpg

        #=========================================================================

    def Type_1(self, num, save=False):
        number = [cv2.resize(number, (136, 320)) for number in self.Number]
        char_n = [cv2.resize(char1, (136, 320)) for char1 in self.Char1]
        hyphen = cv2.resize(self.hyphen, (80, 50))
        Plate = self.plate
        Plate_dimensions = Plate.shape

        Country = cv2.resize(self.country, (380, 100))

        #Country = cv2.resize(self.country, (125, 35))
        Flag = cv2.resize(self.flag, (160, 100))

        for i, Iter in enumerate(range(num)):
            label = ""
            row, col = 160, 60  # row + 83, col + 56

            # Place PERU name on top of the license
            #Plate[8:8+Country_dimensions[0], (Plate_dimensions[1]//2-Country_dimensions[1]//2):(Plate_dimensions[1]//2+Country_dimensions[1]//2) + 1, :] = Country
            Plate[36:36+100, 390:390+380, :] = Country
            # Place Flag on the top right
            Plate[36:36+100, 30:30+160, :] = Flag

            # Char 1
            rand_int = random.randint(0, 9)
            label += self.char_list[rand_int]
            Plate[row:row + 320, col:col + 136, :] = char_n[rand_int]
            col += 156

            # number 2
            rand_int = random.randint(0, 9)
            label += self.char_list[rand_int]
            Plate[row:row + 320, col:col + 136, :] = char_n[rand_int]
            col += 156

            # number 3
            rand_int = random.randint(0, 9)
            label += self.char_list[rand_int]
            Plate[row:row + 320, col:col + 136, :] = char_n[rand_int]
            col += 156

            Plate[row+140:row + 190, col:col + 80, :] = hyphen
            col += 100

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 320, col + 2:col + 136 + 2, :] = number[rand_int]
            col += 156

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 320, col:col + 136, :] = number[rand_int]
            col += 156

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 320, col:col + 136, :] = number[rand_int]
            col += 156

            #Plate = random_bright(Plate)
            if save:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            #else:

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="../CRNN/DB/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=True)
args = parser.parse_args()


img_dir = args.img_dir
A = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

A.Type_1(num_img, save=Save)
print("Type 1 finish")
# A.Type_2(num_img, save=Save)
# print("Type 2 finish")
