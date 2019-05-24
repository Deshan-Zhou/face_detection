from scipy import misc
import pandas as pd
import tensorflow as tf
from face_detect import detect_face
from model import facenet
import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox
import time

resize=160
model= 'C:/Users/Admin/Desktop/基于MTCNN的人脸检测与识别系统/model/20170512-110547/'


def get_feature(image):
    image_results = detection(image)
    if (image_results is None):
        return None
    with tf.Graph().as_default():
        with tf.Session() as sess:
            #导入训练好的用来提取特征模型，即文件夹20170512-110547下面的文件
            facenet.load_model(model)
            #输入图像的占位符
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            #卷积网络最后输出的特征（嵌入）
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            #这个占位符决定现在是不是出于训练阶段
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            #把输入占位符填入相应的值，执行sess.run得到嵌入
            feed_dict = {images_placeholder: image_results, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb

def detection(image):

    minsize = 20                    # 识别的面最小尺寸
    threshold = [0.6, 0.7, 0.7]     # 三个MTCNN网络的阈值
    # 比例项间的比率，用于创建一个扩展的因素金字塔脸大小的检测图像中。用于创建图像中检测到的面部尺寸的比例金字塔的因素
    factor = 0.709
    #获取识别图片的高和宽
    h, w = image.shape[:2]
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            #获得MTCNN的三个神经网络
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        
    #检测图像中的人脸，并为其返回包围框和点。使用的是rgb图像
    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

    img_list = [None]*len(bounding_boxes)
    #未检测到则返回空           
    if len(bounding_boxes) < 1:
        print("can't detect face in the frame")
        return None                                        
    print("检测到%d个人脸\n"% len(bounding_boxes))
    #图像颜色空间转换,显示图像一般用bgr
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(len(bounding_boxes)):
        #得到识别框的4个位置信息
        det = np.squeeze(bounding_boxes[i, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        #将框的左上角坐标，框的高宽进行限定
        margin = 0
        bb[0] = np.maximum(det[0] - margin / 2, 0)          #np.maximum求最大值
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, w)
        bb[3] = np.minimum(det[3] + margin / 2, h)
        # cv2.rectangle 用矩形把找到的脸起来，参数分别是图片，框左上角坐标(x,y)，框高宽，框颜色，款厚度，框线类型，坐标有几位小数
        cv2.rectangle(bgr, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2, 8, 0)
        #裁剪识别的人脸区域，并放缩到卷积神经网络输入大小
        cropped = bgr[bb[1]:bb[3],bb[0]:bb[2],:]
        #cv2.imshow("detected faces", cropped)
        aligned = misc.imresize(cropped,(resize,resize),interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i]= prewhitened
    
    cv2.imshow("detected faces",bgr)  
    image_results = np.stack(img_list)      
    return image_results

def save_future():
    #读取本地视频和打开摄像头
    capture = cv2.VideoCapture(0)
    c=1
    while True:
        ret, frame = capture.read()
        if ret is True:
        
            #cv2.flip 图像翻转，让显示的图像像看镜子一样
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        #cvtColor()颜色空间转换函数
            #特征提取成功
            emb = get_feature(rgb)
            if not(emb is None):
                b = pd.DataFrame(data=emb)
                b.to_csv('future.csv',mode='a',header=False,index=False)
                messagebox.showinfo('操作提示','特征存储成功')
                cv2.destroyAllWindows()
                return None
            
            #大约200个循环，没识别出人脸特征，就关闭摄像头
            timeF = 10
            if(c%timeF == 0):
                messagebox.showinfo('操作提示','没有识别到人脸')
                cv2.destroyAllWindows()     #关闭窗口
                return None
            
            c+=1
            print(c)
            #waitKey()函数的功能是等待一定时间，单位为ms，这里设定会影响到capture.read()多久调用一次
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                return None
        else:
            return None

def check_future():
    #读取本地视频和打开摄像头
    capture = cv2.VideoCapture(0)
    c=1
    while True:
        ret, frame = capture.read()
        if ret is True:
        
            #cv2.flip 图像翻转，让显示的图像像看镜子一样
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        #cvtColor()颜色空间转换函数
            #特征提取成功
            emb = get_feature(rgb)
            if not(emb is None):
                b = pd.read_csv('future.csv')
                future_list = np.array(b)
                #print('emb:',emb,np.shape(emb))
                #print('future:',future_list,np.shape(future_list))
                lens = len(future_list)
                for i in range(lens):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb,future_list[i]))))
                    if dist < 1.21:
                        messagebox.showinfo('操作提示','此人在系统的人脸数据库中')
                        cv2.destroyAllWindows()
                        return None
                
                messagebox.showinfo('操作提示','此人不在系统的人脸数据库中')
                cv2.destroyAllWindows()
                return None
            
            #大约200个循环，没识别出人脸特征，就关闭摄像头
            timeF = 10
            if(c%timeF == 0):
                messagebox.showinfo('操作提示','没有识别到人脸')
                cv2.destroyAllWindows()     #关闭窗口
                return None
            
            c+=1
            print(c)
            #waitKey()函数的功能是等待一定时间，单位为ms，这里设定会影响到capture.read()多久调用一次
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                return None
        else:
            return None

#主界面
win = Tk()
win.title("基于MTCNN的人脸检测与识别系统")
win.minsize(400,200)

recognize_button = Button(win,text="人脸检测与识别",command=check_future)
recognize_button.grid(row=0,column=0)

add_info_button = Button(win,text="采集人脸信息",command=save_future)
add_info_button.grid(row=1,column=0)

win.mainloop()

