import glob
import cv2
import shutil, os
import numpy as np

from matplotlib import pyplot as plt

from random import randint

def blending(pathRgb, pathSmoke):
    src = cv2.imread(pathRgb)
    smoke = cv2.imread(pathSmoke,  cv2.IMREAD_UNCHANGED)
    height, width, depth = src.shape
    # print(smoke.shape)
    smoke = cv2.resize(smoke, (width, height))
    # print(smoke.shape)
    smokeRgb = smoke[:, :, 0:3]
    smokeAlpha = smoke[:, :, 3]

    src = src.astype(float)
    smokeRgb = smokeRgb.astype(float)
    smokeAlpha = smokeAlpha.astype(float) / 255.0
    smokeAlpha = cv2.merge([smokeAlpha, smokeAlpha, smokeAlpha]) 

    # print(smokeAlpha.shape, ' ' , src.shape)
    src = cv2.multiply(1.0 - smokeAlpha, src)
    smokeRgb = cv2.multiply(smokeAlpha, smokeRgb)

    ret = cv2.add(src, smokeRgb)
    ret = ret.astype(np.uint8)
    
    height, width, depth = ret.shape

    width = round(width / 1.5)
    height = round(height / 1.5)

    ret = cv2.resize(ret, (width, height))

    return(ret)

    

smokeImg = '/home/citlab/SmokeData/smokepatent/'
envirImg = '/home/citlab/SmokeData/SUNRGBD-cleanup/SUNRGBD/trainval/rgb/'

fileSmoke = [f for f in glob.glob(smokeImg + "**/*.png", recursive=True)]
fileEnvir = [f for f in glob.glob(envirImg + "**/*.jpg", recursive=True)]

# shutil.copy(fileSmoke[0], '/home/citlab/SmokeData/data/train/neg')
numSmoke = len(fileSmoke)

for num in range(10):
    print(num)
    id1 = num * 4
    id2 = id1 + 1
    id3 = id2 + 1
    id4 = id3 + 1
    x1 = randint(0, numSmoke - 1)
    x2 = randint(0, numSmoke - 1)

    # shutil.copy(fileEnvir[id1], '/home/citlab/SmokeData/data/train/neg')
    # shutil.copy(fileEnvir[id3], '/home/citlab/SmokeData/data/val/neg')
    name1 = '/home/citlab/SmokeData/data/train/pos/' + str(num) + '.png'
    name2 = '/home/citlab/SmokeData/data/val/pos/' + str(num) + '.png'
    src1 = blending(fileEnvir[id2], fileSmoke[x1])
    src2 = blending(fileEnvir[id4], fileSmoke[x2])
    # cv2.imwrite(name1, blending(fileEnvir[id2], fileSmoke[x1]))    
    # cv2.imwrite(name2, blending(fileEnvir[id4], fileSmoke[x2]))    
    # cv2.imshow("tmp1 ", src1)
    # cv2.imshow("tmp2", src2)
    # if (cv2.waitKey() == 'q') :
        # break
    plt.title('RGB image')
    plt.imshow(src1)
    plt.show()
    # print(fileEnvir[id2], ' ', fileSmoke[x1])    
    # print(fileEnvir[id4], ' ', fileSmoke[x2])


# cv2.imwrite('/home/citlab/tmp.png', blending(fileEnvir[0], fileSmoke[0]))