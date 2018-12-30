import numpy as np
import cv2
from sklearn.feature_extraction import image as slicer
from scipy.interpolate import interp1d
'''
imageEnhanced use output as imageED when use with fundus image
'''

def resizeImage(img,shape):
    return cv2.resize(img,shape)
    
def imageEnhanced(img):
    imgE = cv2.addWeighted(src1 = img,alpha = 4, src2 = cv2.GaussianBlur(src=img,ksize=(0,0),sigmaX=5),beta = -4,gamma = 128)   
    mark = np.zeros(img.shape)
    mark = cv2.circle(mark,center=(int(imgE.shape[1]/2),int(imgE.shape[0]/2)),
           radius=int(imgE.shape[1]*0.36),color=(1,1,1),thickness=-1,lineType=8,shift=0)
    imgED = (imgE*mark)+(128*(1-mark))
    return imgE.astype('uint8')

def claheLabChannel(image):
    imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    labs = cv2.split(imageLAB)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16,16))
    labs[0] = clahe.apply(labs[0])
    imageLAB = cv2.merge(labs)
    imageRGB = cv2.cvtColor(imageLAB.astype('uint8'), cv2.COLOR_Lab2RGB)
    return imageRGB

def resize_enhance(img,size=244,patch=10,dim=3,step=10):
    img = cv2.resize(img,(size*10,size*10))
    img = claheLabChannel(img)
    filt = (1/100)*np.ones((10,10))
    convim = cv2.filter2D(img,-1,filt).astype('uint8')
    imgs = slicer.extract_patches(convim, patch_shape=(patch,patch,dim),extraction_step=step)
    imgsr = imgs.reshape(-1,10,10,3)
    imx1 = imgsr.mean(axis=1)
    imx2 = imx1.mean(axis=1)
    imgres = imx2.reshape(size,size,dim)
    return imgres[:,:,1].astype('uint8')

def gradientExtract(img):
    imageGrad = img
    thd_update = 10
    gradient_count = 0
    gradient_pic = np.zeros(imageGrad.shape)
    for i in range(imageGrad.shape[0]):
        previous_val = imageGrad[i,0].astype('float64')
        for j in range(imageGrad.shape[1]):
            val = imageGrad[i,j].astype('float64')
            if np.absolute(previous_val-val)>thd_update:
                if val-previous_val > 0:
                    gradient_count += 1.0
                else:
                    gradient_count -= 1.0
                previous_val = val
            else:
                gradient_count -= 1.0
                if gradient_count<0:
                    gradient_count = 0
            gradient_pic[i,j] =  gradient_count
            if previous_val-val > thd_update and j >=194:
                gradient_count = 0        
        gradient_count = 0

    gradient_count = 0
    gradient_pic1 = np.zeros(imageGrad.shape)
    for i in range(imageGrad.shape[0]-1,-1,-1):
        previous_val = imageGrad[i,imageGrad.shape[1]-1].astype('float64')
        for j in range(imageGrad.shape[1]-1,-1,-1):
            val = imageGrad[i,j].astype('float64')
            if np.absolute(previous_val-val)>thd_update:
                if val-previous_val > 0:
                    gradient_count += 1.0
                else:
                    gradient_count -= 1.0
                previous_val = val
            else:
                gradient_count -= 1.0
                if gradient_count<0:
                    gradient_count = 0

            if previous_val-val > thd_update and j<=30:
                gradient_count = 0
            gradient_pic1[i,j] = gradient_count  

        gradient_count = 0


    gradient_count = 0
    gradient_pic2 = np.zeros(imageGrad.shape)
    for j in range(imageGrad.shape[1]):
        previous_val = imageGrad[0,j].astype('float64')
        for i in range(imageGrad.shape[0]):
            val = imageGrad[i,j].astype('float64')
            if np.absolute(previous_val-val)>thd_update:
                if val-previous_val > 0:
                    gradient_count += 1.0
                else:
                    gradient_count -= 1.0
                previous_val = val

            else:
                gradient_count -= 1.0
                if gradient_count<0:
                    gradient_count = 0

            if i>=194:
                gradient_count = 0            
            gradient_pic2[i,j] = gradient_count  

        gradient_count = 0

    gradient_count = 0
    gradient_pic3 = np.zeros(imageGrad.shape)
    for j in range(imageGrad.shape[1]-1,-1,-1):
        previous_val = imageGrad[imageGrad.shape[0]-1,j].astype('float64')
        for i in range(imageGrad.shape[0]-1,-1,-1):
            val = imageGrad[i,j].astype('float64')
            if np.absolute(previous_val-val)>thd_update:
                if val-previous_val > 0:
                    gradient_count += 1.0
                else:
                    gradient_count -= 1.0
                previous_val = val

            else:
                gradient_count -= 1.0
                if gradient_count<0:
                    gradient_count = 0

            if  i<=30:
                gradient_count = 0            
            gradient_pic3[i,j] =  gradient_count

        gradT = (gradient_pic+gradient_pic1+gradient_pic2+gradient_pic3)/4
        gradCap_t =  interp1d([gradT.min(),gradT.max()],[0,255])
        return gradCap_t(gradT).astype('uint8')
    
def treshold_Enhance(img):
    ret,thresh1 = cv2.threshold(img,40,255,cv2.THRESH_TOZERO)
    thresh1 = cv2.GaussianBlur(thresh1,(5,5),0)
    ret2,thresh2 = cv2.threshold(thresh1,30,255,cv2.THRESH_TOZERO)
    return thresh2

def sobel_Gradient(img):
    sobx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=15)
    soby = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=15)
    alpha = 0.3
    sobxy = alpha*sobx + (1-alpha)*soby 
    sobCapxy =  interp1d([sobxy.min(),sobxy.max()],[0,255])
    sop_imxy = sobCapxy(sobxy)
    claheSopxy = cv2.createCLAHE(clipLimit=1, tileGridSize=(3,3))
    sop_imCLxy = claheSopxy.apply(sop_imxy.astype('uint8'))
    return sop_imCLxy