import numpy as np
import cv2
from matplotlib import pyplot as plt
import Kin
from Rect2D import Rect2D
from Point2D import Point2D
from Joint import Joint


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




kin = Kin.Kin()
kin.load("./dataset/0068/body_and_face.mat")
# kin.getDescriptors()

blocks = np.zeros(((78-35)*200 * (78-35), 50, 4))

for i in range(35, 78):
    rgbImage = cv2.imread("./dataset/0068/rgbReg_frames/00" + str(i) + ".jpg")
    indexImage = cv2.imread("./dataset/0068/bodyIndex_frames/00" + str(i) + ".png")
    indexImage = cv2.cvtColor(indexImage, cv2.COLOR_BGR2GRAY)
    # rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)

    ret, mask = cv2.threshold(indexImage, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img2_fg = cv2.bitwise_and(rgbImage, rgbImage, mask=mask_inv)
    # plt.imshow(img2_fg)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    if(kin._frames[i]._face):
        neckPos = kin._frames[i]._face._boundingBox
    else:
        neckPos = Rect2D(Point2D(0,0), Point2D(0,0))

    headImg = crop(img2_fg, neckPos)

    whiteMaskImg = crop(whiteImage(headImg), neckPos)

    # show(whiteMaskImg)
    # show(mask_inv)

    whiteMaskImg = cv2.bitwise_and(makeMask(whiteMaskImg), mask_inv)

    # show(whiteMaskImg)

    mean = cv2.mean(headImg, mask=whiteMaskImg)
    print mean

    if(kin._frames[i]._body):
        print kin._frames[i]._body._joints[Joint.Head]._orientation
    else:
        print None
#     index = i - 35
#     blocks[index*200:(index+1)*200, 0:200] = mean
#
# show(blocks)