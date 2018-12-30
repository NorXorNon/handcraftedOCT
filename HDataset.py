'''
This file Dataset was designed to read and assign class of images from directory automantically
method - TestTrainList is used for extract training and testing data in format of x_train, y_train and x_test y_test
       - Toimage is designed to convert batch of path of images to batch of images
       - batch_generator is iterative function for feeding batches of images to iterative fitting function
       - _Kfold is to create cross validation generator
       - batch_fit is created for manual loop for fitting the data
'''
from os import listdir
from os.path import join
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from numpy.random import permutation
from sklearn.model_selection import KFold
import glob
import HPreprocess as Preprocess
from scipy.interpolate import interp1d
from tqdm import tqdm

def TestTrainList(path,test_size,random_state,shuffle):
    imRawDict = dict()
    imTrainTuple = list()
    imTestTuple = list()
    mapping = dict()
    
    mainPath = glob.glob(path+'*')
    classList = list()
    for i, word in enumerate(mainPath):
        c = word.replace(path,'')
        classList.append(c)        
    imRawDict = dict().fromkeys(classList)  
    
    numberOfClass = len(classList)
    for m ,classL in enumerate(classList):
        mapping[classL] = m
    
    for i, word in enumerate(classList):
        imlist = glob.glob(path+word+'/*.jpeg')
        imRawDict[word] = imlist
    
    for classInClassList in classList:
            X_train, X_test = train_test_split(
                imRawDict[classInClassList], test_size = test_size, random_state = random_state,shuffle=shuffle)
            imTrainTuple += set(zip([mapping[classInClassList]]*len(X_train),X_train))            
            imTestTuple += set(zip([mapping[classInClassList]]*len(X_test),X_test))
    imTrainTuple = permutation(imTrainTuple)
    imTestTuple = permutation(imTestTuple)       
    return mapping,np.array(imTrainTuple) , np.array(imTestTuple)

def batch_generator(data,batch_size,shape,shuffleIm,num_class):
    if shuffleIm:
        data = permutation(data)
    label, path = zip(*data)
    path = np.array(path)
    label = np.array(label)
    while True:
        for index in range(0,label.shape[0],batch_size):            
            batch_x = path[index:min(index+batch_size,path.shape[0])]
            batch_x = ToImage(batch_x,shape)
            batch_y = label[index:min(index+batch_size,label.shape[0])]
            batch_y = to_categorical(batch_y,num_class)    
            yield np.array(batch_x) , np.array(batch_y)
        
def ToImage(batch_input,shape):
    batch = list()
    for path in tqdm(batch_input):
        img = cv2.imread(path,0)
        #img = Preprocess.claheLabChannel(img)
        img = cv2.resize(img,shape)
        #img = Preprocess.treshold_Enhance(img)
        #img = Preprocess.sobel_Gradient(img)
        #img = Preprocess.resize_enhance(img)
        #img = Preprocess.gradientExtract(img)
        #img = Preprocess.claheLabChannel(img)
        #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        #img = img.astype('float32')
        #img = cv2.Canny(img,250,120)
        #img = Preprocess.imageEnhanced(img)
        batch.append(img)
    batch = np.array(batch)
    '''
    try:
        batch = batch_normalise_3d(batch)
    except:
        batch = batch_normalise_1d(batch)
    '''
    return np.array(batch)

def batch_normalise_3d(batch):
    for i in range(batch.shape[0]):
        batch[i,:,:,0] = (batch[i,:,:,0]-np.mean(batch[i,:,:,0]))/(np.sqrt(batch[i,:,:,0].var()+ 1e-5))
        batch[i,:,:,1] = (batch[i,:,:,1]-np.mean(batch[i,:,:,1]))/(np.sqrt(batch[i,:,:,1].var()+ 1e-5))
        batch[i,:,:,2] = (batch[i,:,:,2]-np.mean(batch[i,:,:,2]))/(np.sqrt(batch[i,:,:,2].var()+ 1e-5))
    return batch

def batch_normalise_1d(batch):
    for i in range(batch.shape[0]):
        #batch[i,:,:] = (batch[i,:,:]-np.mean(batch[i,:,:]))    /(np.sqrt(batch[i,:,:].var()+ 1e-5))
        m = interp1d([batch[i,:,:].min(),batch[i,:,:].max()],[0,1])
        batch[i,:,:] = m(batch[i,:,:])
    return batch

def _Kfold(data,nFold):      
    kf = KFold(n_splits=nFold)
    data = permutation(data)
    for indTrain, indValidate in kf.split(data):
            dataT = data[indTrain]
            dataV = data[indValidate]
            yield dataT , dataV

            

def batch_fit(data,batch_size,shape,shuffleIm,num_class):
    
    if shuffleIm:
        data = permutation(data)
    label, path = zip(*data)
    path = np.array(path)
    label = np.array(label)

    for index in range(0,label.shape[0],batch_size):            
        batch_x = path[index:min(index+batch_size,path.shape[0])]
        batch_x = ToImage(batch_x,shape)
        batch_y = label[index:min(index+batch_size,label.shape[0])]
        batch_y = to_categorical(batch_y,num_class)    
        yield np.array(batch_x) , np.array(batch_y)

def single_fit(data,shape,shuffleIm,num_class):
    if shuffleIm:
        data = permutation(data)
    label, path = zip(*data)
    path = np.array(path)
    label = np.array(label)         
    batch_x = ToImage(path,shape)
    batch_y = to_categorical(label,num_class)    
    return np.array(batch_x) , np.array(batch_y)

def data_to_txt(name,data):
    with open(name,'w') as text_files:        
        for label,path in data:
            text_files.write(label+','+path+'\n')
      
def txt_to_var(path):
    dataList = list()
    with open(path) as text_file:
        data = text_file.readlines()
    for txt in data:
        label,path = txt.split(',')
        dataList.append([label,path[:-1]])
    return np.array(dataList)

def txt_tofold_var(path):
    dataList = list()
    with open(path) as linkToPatient:
        links = linkToPatient.readlines()
        links = [i[:-1] for i in links]
    for item in links:
        if not item.split('/')[-1].startswith('.'):
            with open(item) as patient_file:
                dataPatitentPath = patient_file.readlines()
                dataPatitentPath = [i[:-1] for i in dataPatitentPath]
            for txt in dataPatitentPath:
                if not txt.split('/')[-1].startswith('.'):
                    label =  txt.split('/')[-1].split('-')[0]
                    if label == 'NORMAL':
                        label = 0
                    elif label == 'CNV':
                        label = 1
                    elif label == 'DME':
                        label = 2
                    elif label == 'DRUSEN':
                        label = 3       
                dataList.append([label,txt])
    return np.array(dataList)