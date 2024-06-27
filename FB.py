import cv2
import numpy as np
import asyncio
import time

def chek_second(filt_img, perim_img, trash):  # ???????? ?? ???? ?????????? ?? ????????
    sq_filt =sum(list(map(lambda x: sum(x), filt_img)))
    sq_per = sum(list(map(lambda x: sum(x), perim_img)))
    koeff = sq_filt / sq_per
    trash = sum(list(map(lambda x: sum(x), perim_img)))
    if koeff>22:
        return True
    else:
        print(koeff)
        return False

def chek_first(filt_img,perim_img):   #???????? ?? ???? ?????????? ?? ????????
    sq_filt = sum(sum(filt_img))
    sq_per = sum(sum(perim_img))
    koeff = sq_filt/sq_per
    word = ''
    result = 0
    return  sq_filt, sq_per, koeff

def activator_test(img, x):
    bg_min = np.array((60), np.uint8)
    bg_max = np.array((255), np.uint8)
    img = cv2.inRange(img, bg_min, bg_max)
    img = img[:, x-1:x+1]
    img = list(map(lambda x: x[0], img))
    return sum(img)>1000

def activator_1(img):
    bg_min = np.array((60), np.uint8)
    bg_max = np.array((255), np.uint8)
    x = 100
    img = cv2.inRange(img, bg_min, bg_max)
    if sum(list(map(lambda x: x[0], img[:, x - 1:x + 1]))) >1000 and sum(list(map(lambda x: x[0], img[:, 68:70]))) < 1000 and sum(list(map(lambda x: x[0], img[:, 40:42]))) < 1000:
        return True
    else:
        return False
        
        
def activator_2(img):
    bg_min = np.array((60), np.uint8)
    bg_max = np.array((255), np.uint8)
    x = 40
    img = cv2.inRange(img, bg_min, bg_max)
    if sum(list(map(lambda x: x[0], img[:, x - 1:x + 1]))) >1000 and sum(list(map(lambda x: x[0], img[:, 25:27]))) < 1000 and sum(list(map(lambda x: x[0], img[:, 1:3]))) < 1000:
        return True
    else:
        return False
        
def activator_3(img):
    bg_min = np.array((60), np.uint8)
    bg_max = np.array((255), np.uint8)
    x = 20
    img = cv2.inRange(img, bg_min, bg_max)
    if sum(list(map(lambda x: x[0], img[:, x - 1:x + 1]))) >1000:# and sum(list(map(lambda x: x[0], img[:, 8:10]))) < 1000:
        return True
    else:
        return False
    
    
def test(clean_img):
    gray = clean_img
#    gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=5, sigmaY=2)
 #   gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=5, sigmaY=2)
  #  gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=5, sigmaY=2)
   # gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=5, sigmaY=2)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Close contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    sq_per1 = sum(list(map(lambda x: sum(x), close)))
    # Find outer contour and fill with white
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(close, cnts, [255, 255, 255])
    sq_per = sum(list(map(lambda x: sum(x), close)))
    if sq_per1 - sq_per >60000:
        return False
    else:
        print(sq_per1 - sq_per)
        return True
    
def video_obrabotka(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=2, sigmaY=2)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=2, sigmaY=2)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=2, sigmaY=2)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=2, sigmaY=2)
    bg = img
    clean_img = img
    bg_min = np.array((60), np.uint8)
    bg_max = np.array((255), np.uint8)
    monochrome_img = cv2.inRange(bg, bg_min, bg_max)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    con, hir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # risovanie konturov
    contour_img = np.zeros(bg.shape, dtype='uint8')  # ????? ???????# ????? ???????
    cv2.drawContours(contour_img, con, -1, (230, 111, 148), 1)  # (???????, ?????? ?????????, ???????, ????, ???????)
    return monochrome_img, thresh, contour_img, clean_img

