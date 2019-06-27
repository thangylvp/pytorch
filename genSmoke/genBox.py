import glob
import cv2
import shutil, os
import numpy as np
import random
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET 

from random import randint
from multiprocessing import Pool

def genListBg(rgbPath) : 
    ret = []
    cnt = 0
    for tmp in rgbPath:
        # tmp = rgbPath[idx]
        src = cv2.imread(tmp)
        h, w, d = src.shape

        if (max(h, w) >= 500) and (h * w > 500 * 250) and (max(h, w) / min(h, w) <  2.1):
            cnt = cnt + 1
            print(tmp)
            ret.append(tmp)

        if (cnt == 2000):
            break

    with open('a.txt', 'w') as f:
        for tmp in ret:
            f.write("%s\n" % tmp)

    return ret


def loadListBgImg(path):
    ret = [line.rstrip('\n') for line in open(path)]
    return ret

smokeImg = '/home/citlab/SmokeData/smokepatent/'
envirImg = '/home/citlab/SmokeData/SUNRGBD-cleanup/SUNRGBD/trainval/rgb/'
outRgb = '/home/citlab/SmokeData/rcnn/rgb/'
outXml = '/home/citlab/SmokeData/rcnn/anno/'

fileSmoke = [f for f in glob.glob(smokeImg + "**/*.png", recursive=True)]
fileEnvir = [f for f in glob.glob(envirImg + "**/*.jpg", recursive=True)]

numSmoke = len(fileSmoke)
# listRgb = genListBg(fileEnvir)
listRgb = loadListBgImg("a.txt")


def randomPosition(rgbW, rgbH, cropW, cropH) :
    # print(rgbW, rgbH, cropW, cropH )
    return random.randint(0, rgbW - cropW), random.randint(0, rgbH - cropH)


def blending(src, smokeCrop):
    rgbH, rgbW, rgbD = src.shape
    cropH, cropW, cropD = smokeCrop.shape
    # print("smoke size : ", smokeCrop.shape)
    x, y = randomPosition(rgbW, rgbH, cropW, cropH)

    tmp = np.zeros((src.shape[0], src.shape[1], 4), np.uint8)

    tmp[y : y + cropH, x : x + cropW, : ] = smokeCrop

    smokeRgb = tmp[:, :, 0:3]
    smokeAlpha = tmp[:, :, 3]

    src = src.astype(float)
    smokeRgb = smokeRgb.astype(float)
    smokeAlpha = smokeAlpha.astype(float) / 255.0
    smokeAlpha = cv2.merge([smokeAlpha, smokeAlpha, smokeAlpha]) 

    src = cv2.multiply(1.0 - smokeAlpha, src)
    smokeRgb = cv2.multiply(smokeAlpha, smokeRgb)

    ret = cv2.add(src, smokeRgb)
    ret = ret.astype(np.uint8)
    
    # cv2.imwrite("0001.png", ret)
    # genXml("0001.png", (rgbH, rgbW), (x, y, cropW, cropH))

    return ret, (x, y, cropW, cropH)

    
def genXml(path, fileName, imgSize, boxS):
    # print(boxS, type(boxS), boxS[0] + boxS[2])
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
    

    doc.write(path) 
    return


def process(p):
    # print(p)
    bgImg = cv2.imread(listRgb[p], 1)
    # cv2.imshow("tmp", bgImg)
    # cv2.waitKey(1000)
    # print(type(bgImg))
    idSmoke = randint(0, numSmoke - 1)
    skImg = cv2.imread(fileSmoke[idSmoke], cv2.IMREAD_UNCHANGED)

    alpha = skImg[:, :, 3]

    tmp = (alpha > 0).astype(np.uint8)

    rect = cv2.boundingRect(alpha)     

    h, w, d = bgImg.shape
    nw = 0
    nh = 0
    if (h == max(h, w)) :
        nw = round(w / (h / 500.0))
        nh = 500
    else :
        nh = round(h / (w / 500.0))
        nw = 500

    rgb = cv2.resize(bgImg, (nw, nh))
    rgbH, rgbW, rgbD = rgb.shape
    crop = skImg[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2], :]
    smkH, smkW, smkD = crop.shape
    if (smkH > rgbH) :
        ratio = smkH / rgbH
        ratio += 0.01
        ratio = 1.0 / ratio
        crop = cv2.resize(crop, None, fx = ratio, fy = ratio)
        smkH, smkW, smkD = crop.shape

    if (smkW > rgbW) : 
        ratio = smkW / rgbW
        ratio += 0.01
        ratio = 1.0 / ratio
        crop = cv2.resize(crop, None, fx = ratio, fy = ratio)
        smkH, smkW, smkD = crop.shape

    ratio = (smkW * smkH) / (rgbW * rgbH)
    if (ratio > 0.8):
        # print('zzzz')
        tmp = randint(2000, 4000)    
        newRatio = tmp / 10000.0
        newRatio = np.sqrt(newRatio)
        ratio = newRatio / ratio

        crop = cv2.resize(crop, None, fx = ratio, fy = ratio)
        smkH, smkW, smkD = crop.shape
    
    outImg, box = blending(rgb, crop)
    # cv2.imshow("tmp", outImg)
    # cv2.waitKey(1000)
    outName = str(p).zfill(4)
    outRgbName = outName + '.png'
    outRgbPath = os.path.join(outRgb, outRgbName)
    cv2.imwrite(outRgbPath, outImg)

    outXmlName = outName + '.xml'
    outXmlPath = os.path.join(outXml, outXmlName)
    genXml(outXmlPath, outRgbName, outImg.shape, box)
    print("Done ", p)
    if (max(box[2], box[3]) / min(box[2], box[3]) > 1.98) : 
        print(" Ratio : ", max(box[2], box[3]) / min(box[2], box[3]))
    


# shutil.copy(fileSmoke[0], '/home/citlab/SmokeData/data/train/neg')


# blending(rgb, crop)

if __name__ == "__main__":    
    numBg = list(range(len(listRgb)))
    
#    for p in numBg:
#        process(p)

    p = Pool(8)
    p.map(process, numBg)
    p.close()
    p.join()
    