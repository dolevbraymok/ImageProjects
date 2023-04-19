import cv2
import numpy as np
import argparse



def func():
    image = cv2.imread("Images/kiki_window.jpg")
    (h,w,c)=image.shape[:3]
    print("width: {} pixels".format(w))
    print("height: {} pixels".format(h))
    print("channels: {} pixels".format(c))
    
if __name__ == '__main__':
    func()