import glob
import cv2
import shutil, os
import numpy as np
import random
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET 

from random import randint

def randomPosition(rgbW, rgbH, cropW, cropH) :
    print(rgbW, rgbH, cropW, cropH )
    return random.randint(0, rgbW - cropW), random.randint(0, rgbH - cropH)

def blending(src, smokeCrop):
    # src = cv2.imread(pathRgb)
    # smoke = cv2.imread(pathSmoke,  cv2.IMREAD_UNCHANGED)
    rgbH, rgbW, rgbD = src.shape
    # print(smoke.shape)
    cropH, cropW, cropD = smokeCrop.shape
    # print(smoke.shape)

    x, y = randomPosition(rgbW, rgbH, cropW, cropH)

    tmp = np.zeros((src.shape[0], src.shape[1], 4), np.uint8)

    tmp[y : y + cropH, x : x + cropW, : ] = crop
    # plt.imshow(tmp)
    # plt.show()
    
    smokeRgb = tmp[:, :, 0:3]
    smokeAlpha = tmp[:, :, 3]

    src = src.astype(float)
    smokeRgb = smokeRgb.astype(float)
    smokeAlpha = smokeAlpha.astype(float) / 255.0
    smokeAlpha = cv2.merge([smokeAlpha, smokeAlpha, smokeAlpha]) 

    # print(smokeAlpha.shape, ' ' , src.shape)
    src = cv2.multiply(1.0 - smokeAlpha, src)
    smokeRgb = cv2.multiply(smokeAlpha, smokeRgb)

    ret = cv2.add(src, smokeRgb)
    ret = ret.astype(np.uint8)
    
    # height, width, depth = ret.shape

    # width = round(width / 1.5)
    # height = round(height / 1.5)

    # ret = cv2.resize(ret, (width, height))
    cv2.imwrite("0001.png", ret)
    genXml("0001.png", (rgbH, rgbW), (x, y, cropW, cropH))

    
def genXml(fileName, imgSize, boxS):
    print(boxS, type(boxS), boxS[0] + boxS[2])
    doc = ET.parse('000002.xml')
    
    size = doc.findall('size') 
    size[0].find('width').text = str(imgSize[1])
    size[0].find('height').text = str(imgSize[0])
    
    name = doc.findall('filename')
    name[0].text = fileName
    
    objs = doc.findall('object')    
    box = objs[0].find('bndbox')    
    xx = round(boxS[0] + boxS[2])
    yy = round(boxS[1] + boxS[3])
    box.find('xmin').text = str(boxS[0])
    box.find('ymin').text = str(boxS[1])
    box.find('xmax').text = str(xx)
    box.find('ymax').text = str(yy)
    
    doc.write('test.xml') 
    return

smokeImg = '/home/thangnv/data/smoke'
envirImg = '/home/thangnv/data/rgb'

fileSmoke = [f for f in glob.glob(smokeImg + "**/*.png", recursive=True)]
fileEnvir = [f for f in glob.glob(envirImg + "**/*.jpg", recursive=True)]

# shutil.copy(fileSmoke[0], '/home/citlab/SmokeData/data/train/neg')
numSmoke = len(fileSmoke)

rgbPath = fileEnvir[0]
skePath = fileSmoke[0]

print(rgbPath)
print(skePath)

rgb = cv2.imread(rgbPath)
smoke = cv2.imread(skePath, cv2.IMREAD_UNCHANGED)

# cv2.imshow("rgb", rgb)
# cv2.imshow("smoke", smoke)

alpha = smoke[:, :, 3]

tmp = (alpha > 0).astype(np.uint8)

rect = cv2.boundingRect(alpha)     

h, w, d = rgb.shape
nh = 0
nw = 0

if (max(h, w) >= 500) :
    if (h == max(h, w)) :
        nw = round(w / (h / 500.0))
        nh = 500
    else :
        nh = round(h / (w / 500.0))
        nw = 500

rgb = cv2.resize(rgb, (nw, nh))
h, w, d = rgb.shape
print("new size : ", h, w, d)


crop = smoke[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2], :]
crop = cv2.resize(crop, (round(crop.shape[1] / 2), round(crop.shape[0] / 2)))

# cv2.imshow("crop", crop)
# cv2.waitKey()

# plt.imshow(crop)
# plt.show()

blending(rgb, crop)