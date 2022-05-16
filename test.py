import datetime #时间戳
import os
#import pyttsx3 #文字转语音
import time

import autoit #windiws 自动化ui操作
import cv2
import cv2 as cv
import numpy as np #数据科学计算，加快视觉计算速度
import pymysql #mysql连接库，不解释
import pyperclip #读取复制黏贴缓冲区
from PIL import ImageGrab #Python图像处理库PIL
import traceback #可以打印出捕获的错误的具体错误堆栈，便于调试
def dingwei(image_muban):
    '''
    此函数的功能是通过传入的图片用opencv定位图片所在的位置，返回定位结果
    计算机视觉opencv 定位模块，具体技术请看opencv 文档
    '''
    im = ImageGrab.grab()
    imm = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    imm = cv2.cvtColor(imm, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("image",imm)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    img_rgb = imm
    template = cv2.imread(image_muban, 0)
    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return (min_val, max_val, min_loc, max_loc)

def tihuanbaise():
    #用于替换白色头像颜色为黑色
    share=ImageGrab.grab()  #截图
    share=cv2.cvtColor(np.array(share), cv2.COLOR_RGB2BGR)
    share = cv.cvtColor(share, cv.COLOR_BGR2GRAY)
    #cv2.imshow("ybk1",share) #在窗口中显示图片
    width,height =share.shape #高和宽的高度为图像高宽
    B=(share) #通道分离
    for i in range(0,width):
        for j in range(height):
                if B[i,j]>245:
                    share.itemset((i,j),0) #修改图像的像素值
    ret, thresh = cv.threshold(share, 244, 255,0)
    return thresh

def dingweituxiang(thresh):
    #定位聊天头像
    img_rgb=thresh
    template = cv.imread('touxiang.png',0)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_rgb,template,cv.TM_CCOEFF_NORMED)
    threshold = 0.9 #定义匹配阈值
    loc = np.where(res >= threshold)
    oldpt=-50 #定位上一个图像横坐标的初始值
    tubiao=[] #定义返回的图标数组
    for pt in zip(*loc[::-1]):
        if pt[1]-oldpt>50 :
            print(w,h)
            tubiao.append(pt)
            cv.rectangle(img_rgb, pt, (pt[0] + w-24, pt[1] + h-24), (0,0,255), 2)
            #print("右键位置 横坐标"+str(pt[0] + w+12)+" 纵坐标"+str(pt[1] + h))
            #print("at右键位置 横坐标"+str(pt[0] + w-24)+" 纵坐标"+str(pt[1] + h-24))
        oldpt = pt[1]
    return tubiao
def pipeidingwei():
    '''复制定位并读取@机器人的信息'''
    tubiao=(dingweituxiang(tihuanbaise()))
    print(tubiao)
    autoit.mouse_click('right', tubiao[-1][0]+95-24,tubiao[-1][1]+52-24,speed=0)
    autoit.mouse_click('left', tubiao[-1][0]+95-24+10,tubiao[-1][1]+52-24+10,speed=0)
    print(tubiao[-1][0]+95-24+10,tubiao[-1][1]+52-24+10)
    autoit.mouse_click('right', tubiao[-1][0]+95+12,tubiao[-1][1]+52,speed=0)
    min_val, max_val, min_loc, max_loc = dingwei('shan_chu_anniu.png')
    cishu=1
    while max_val<=0.95 and cishu<=3:
        min_val, max_val, min_loc, max_loc = dingwei('shan_chu_anniu.png')
        cishu=cishu+1
    min_val, max_val, min_loc, max_loc = dingwei('fu_zhi_anniu.png')
    if max_val >= 0.95:
        autoit.mouse_click('left', max_loc[0]+50, max_loc[1]+20,speed=0)
    else:
        min_val, max_val, min_loc, max_loc = dingwei('shan_chu_anniu.png')
        autoit.mouse_click('left', max_loc[0] + 50, max_loc[1] + 20, speed=0)
        min_val, max_val, min_loc, max_loc = dingwei('shanchu_gaitiao_xinxi.png')
        cishu = 1
        while max_val<=0.95 and cishu<=3:
            min_val, max_val, min_loc, max_loc = dingwei('shanchu_gaitiao_xinxi.png')
            cishu = cishu+1
        autoit.mouse_click('left', max_loc[0] + 230, max_loc[1] + 180, speed=0)
    print(pyperclip.paste())
    if "工作提醒" in pyperclip.paste() and "hello ai" in pyperclip.paste():
        pyperclip.copy('已收到您的信息')
        min_val, max_val, min_loc, max_loc = dingwei("fasong.png")
        max_los_send_with = max_loc[0] + 80
        max_los_send_heigh = max_loc[1] - 15
        autoit.mouse_click('left', max_los_send_with, max_los_send_heigh, speed=0)
        autoit.send("^v")


        autoit.mouse_click('right', tubiao[-1][0] + 95 + 12, tubiao[-1][1] + 52, speed=0)
        min_val, max_val, min_loc, max_loc = dingwei('shan_chu_anniu.png')
        cishu = 1
        while max_val <= 0.95 and cishu <= 3:
            min_val, max_val, min_loc, max_loc = dingwei('shan_chu_anniu.png')
            cishu = cishu + 1
        autoit.mouse_click('left', max_loc[0] + 50, max_loc[1] + 20, speed=0)
        min_val, max_val, min_loc, max_loc = dingwei('shanchu_gaitiao_xinxi.png')
        cishu = 1
        while max_val<=0.95 and cishu<=3:
            min_val, max_val, min_loc, max_loc = dingwei('shanchu_gaitiao_xinxi.png')
            cishu = cishu+1
        autoit.mouse_click('left', max_loc[0] + 230, max_loc[1] + 180, speed=0)
        autoit.mouse_click('left', max_los_send_with, max_los_send_heigh, speed=0)
        autoit.send("{ENTER}")
    else:
        min_val, max_val, min_loc, max_loc = dingwei("fasong.png")
        max_los_send_with = max_loc[0] + 80
        max_los_send_heigh = max_loc[1] - 15
        autoit.mouse_click('left', max_los_send_with, max_los_send_heigh, speed=0)
        autoit.send("{Backspace}")
        autoit.mouse_click('right', tubiao[-1][0] + 95 + 12, tubiao[-1][1] + 52, speed=0)
        min_val, max_val, min_loc, max_loc = dingwei('shan_chu_anniu.png')
        cishu = 1
        while max_val <= 0.95 and cishu <= 3:
            min_val, max_val, min_loc, max_loc = dingwei('shan_chu_anniu.png')
            cishu = cishu + 1
        autoit.mouse_click('left', max_loc[0] + 50, max_loc[1] + 20, speed=0)
        min_val, max_val, min_loc, max_loc = dingwei('shanchu_gaitiao_xinxi.png')
        cishu = 1
        while max_val<=0.95 and cishu<=3:
            min_val, max_val, min_loc, max_loc = dingwei('shanchu_gaitiao_xinxi.png')
            cishu = cishu+1
        autoit.mouse_click('left', max_loc[0] + 230, max_loc[1] + 180, speed=0)
    pyperclip.copy('')
def xunzhaoatdequn():
    min_val, max_val, min_loc, max_loc = dingwei('have_people_at_me.png')
    if max_val>=0.95:
        autoit.mouse_click('left', max_loc[0],max_loc[1],speed=0)
    while max_val>=0.95:
        min_val, max_val, min_loc, max_loc = dingwei('ti_dao_le_ni.png')
        cishu=1
        while max_val<0.95 and cishu<=3:
            min_val, max_val, min_loc, max_loc = dingwei('ti_dao_le_ni.png')
            cishu = cishu + 1
        if max_val>=0.95:
            autoit.mouse_click('left', max_loc[0], max_loc[1], speed=0)
            pipeidingwei()
        if cishu>3:
            return "loss ti_dao_le_ni"
xunzhaoatdequn()