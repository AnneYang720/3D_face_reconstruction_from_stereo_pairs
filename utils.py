import numpy as np
import cv2
from sklearn.cluster import KMeans

# function [ foreground ] = face_mask_extraction( I , which_image )
def face_mask_extraction(img):
    #将Opencv中图像默认BGR转换为通用的RGB格式
    img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    (height, width, channels) = img.shape
    #将图像数据转换为需要进行Kmeans聚类的Data
    img_data = img.reshape(height*width, channels)

    print( '[INFO] Kmeans 颜色聚类......' )
    #调用sklearn中Kmeans函数
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(img_data)
    pixel_labels = kmeans.labels_

    # Labels
    label_back = round(np.mean(pixel_labels[0:10,0:10]))
    labeled_pixels = (pixel_labels!=label_back) #不等于是true，等于是false

    # Delete small details
    kernel = np.ones((5,5),np.uint8)
    labeled_pixels = cv2.morphologyEx(labeled_pixels, cv2.MORPH_OPEN, kernel) #去掉背景中的噪点
    labeled_pixels = cv2.morphologyEx(labeled_pixels, cv2.MORPH_CLOSE, kernel) #去掉前景中的噪点

    return labeled_pixels