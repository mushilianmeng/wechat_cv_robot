from matplotlib import pyplot as plt
import cv2
import cv2 as cv
import numpy as np
def ShowImage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)  # 等待时间，0表示任意键退出
    cv2.destroyAllWindows()

img_rgb = cv.imread('../test.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('../1test.png',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.45
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
ShowImage('image',img_rgb)