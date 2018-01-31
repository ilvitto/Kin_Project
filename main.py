import numpy as np
import cv2
from matplotlib import pyplot as plt
import Kin
import sys


def blackImage(img):
    return np.zeros(img.shape, np.uint8)


def whiteImage(img):
    return cv2.bitwise_not(blackImage(img))


def makeMask(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop(img, rect):
    result = blackImage(img)
    result[rect._lu._y:rect._rb._y, rect._lu._x:rect._rb._x] = img[rect._lu._y:rect._rb._y,
                                                               rect._lu._x:rect._rb._x]
    return result


def show(img):
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

dataset1 = "0064"
dataset2 = "0068"
dataset3 = "0070"
dataset4 = "0065"

#70 - 68 - 64
if len(sys.argv) > 1:
    dataset_number = sys.argv[1]
    kin1 = Kin.Kin()
    desc1, classification1 = kin1.run(dataset_number)
else:
    print "Choose a dataset!"

# kin2 = Kin.Kin()
# desc2, classification2 = kin2.run(dataset2)
#
# kin3 = Kin.Kin()
# classification3 = kin3.classify(desc1 + desc2, method=Kin.GAP)
#
# print classification1.labels_
#
# print classification2.labels_
# print classification3.labels_
# TODO: Kmeans complessivo oltre che incrementale

# kin.load("./dataset/0064/body_and_face.mat")
# kin.getDescriptors()

# blocks = np.zeros(((78-35)*200 * (78-35), 50, 4))
#
# for i in range(1, len(kin._frames)):
#     if i < 10:
#         str_i = "000"+str(i)
#     elif i < 100:
#         str_i = "00"+str(i)
#     elif i < 1000:
#         str_i = "0" + str(i)
#     else:
#         str_i = str(i)
#     rgbImage = cv2.imread("./dataset/0068/rgbReg_frames/" + str_i + ".jpg")
#     indexImage = cv2.imread("./dataset/0068/bodyIndex_frames/" + str_i + ".png")
#     indexImage = cv2.cvtColor(indexImage, cv2.COLOR_BGR2GRAY)
#     rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)
#
#     ret, mask = cv2.threshold(indexImage, 10, 255, cv2.THRESH_BINARY)
#     mask_inv = cv2.bitwise_not(mask)
#
#     img2_fg = cv2.bitwise_and(rgbImage, rgbImage, mask=mask_inv)
#
#     if(kin._frames[i]._face):
#         neckPos = kin._frames[i]._face._boundingBox
#     else:
#         neckPos = Rect2D(Point2D(0,0), Point2D(0,0))
#
#     headImg = crop(img2_fg, neckPos)
#
#     whiteMaskImg = crop(whiteImage(headImg), neckPos)
#
#     # show(whiteMaskImg)
#     # show(mask_inv)
#
#     whiteMaskImg = cv2.bitwise_and(makeMask(whiteMaskImg), mask_inv)
#
#     # show(headImg)
#
#     mean = cv2.mean(headImg, mask=whiteMaskImg)
#     if sum(mean) != 0:
#
#         print i, "RGB", mean,
#     else:
#         print i, "None",
#
#     if(kin._frames[i]._faceHD):
#         #print kin._frames[i]._body._joints[Joint.Head]._orientation
#         print "ROT", kin._frames[i]._faceHD._rotation.item()
#     else:
#         print None
#     index = i - 35
#     blocks[index*200:(index+1)*200, 0:200] = mean
#
# show(blocks)
