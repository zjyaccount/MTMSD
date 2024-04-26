import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len



class HSIDataLoader(object):
    def __init__(self, param={}) -> None:
        self.data_param = param.get('data', {})
        self.data = None #原始读入X数据 shape=(h,w,c)
        self.labels = None #原始读入Y数据 shape=(h,w,1)

        # 参数设置
        self.data_sign = self.data_param.get('data_sign', 'Indian')
        self.patch_size = self.data_param.get('patch_size', 32) # n * n
        self.padding = self.data_param.get('padding', True) # n * n
        self.remove_zeros = self.data_param.get('remove_zeros', False)
        self.batch_size = self.data_param.get('batch_size', 256)
        self.select_spectral = self.data_param.get('select_spectral', []) # [] all spectral selected

        self.squzze = True

        self.split_row = 0
        self.split_col = 0

        self.light_split_ori_shape = None
        self.light_split_map = [] 



    def load_data(self):
        data, labels = None, None
        if self.data_sign == "Indian_Pines":
            data = sio.loadmat('./datasets/indian_pines_16/Indian_pines_corrected.mat')['indian_pines_corrected']
            labels = sio.loadmat('./datasets/indian_pines_16/Indian_pines_gt.mat')['indian_pines_gt']
        elif self.data_sign == "PaviaU":
            data = sio.loadmat('./datasets/PaviaU_9/PaviaU.mat')['paviaU']
            labels = sio.loadmat('./datasets/PaviaU_9/PaviaU_gt.mat')['paviaU_gt']
        elif self.data_sign == "Houston2018":
            data = sio.loadmat('./datasets/Houston2018/Houston2018.mat')['houston2018']
            labels = sio.loadmat('./datasets/Houston2018/Houston2018_gt.mat')['houston2018_gt']
        elif self.data_sign == "Salinas":
            data = sio.loadmat('../data/Salinas_corrected.mat')['salinas_corrected']
            labels = sio.loadmat('../data/Salinas_gt.mat')['salinas_gt']
        elif self.data_sign == "Longkou":
            data = sio.loadmat('./datasets/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
            labels = sio.loadmat('./datasets/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
        else:
            pass
        print("ori data load shape is", data.shape, labels.shape)
        if len(self.select_spectral) > 0:  #user choose spectral himself
            data = data[:,:,self.select_spectral]
        return data, labels

    def get_ori_data(self):
        return np.transpose(self.data, (2,0,1)), self.labels

    def padWithZeros_even(self, X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin - 1, X.shape[1] + 2 * margin - 1, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    def createImageCubes_even(self, X, y, windowSize=32, removeZeroLabels = True):
        margin = windowSize // 2
        zeroPaddedX = self.padWithZeros_even(X, margin=margin)
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin + 1):
            for c in range(margin, zeroPaddedX.shape[1] - margin + 1):
                patch = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            patchesData = patchesData[patchesLabels>0,:,:,:]
            patchesLabels = patchesLabels[patchesLabels>0]
            patchesLabels -= 1

        return patchesData, patchesLabels

    def _padding(self, X, margin=2):
        # pading with zeros
        w,h,c = X.shape
        new_x, new_h, new_c = w+margin*2, h+margin*2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x+w, start_y:start_y+h,:] = X
        return returnX

    def get_patches_by_light_split(self, X, Y, patch_size=1):
        h, w, c = X.shape
        row = h // patch_size
        if h % patch_size != 0:
            row += 1
        col = w // patch_size
        if w % patch_size != 0:
            col += 1
        res = np.zeros((row*col, patch_size, patch_size, c))
        self.light_split_ori_shape = X.shape
        resY = np.zeros((row*col))
        index = 0
        for i in range(row):
            for j in range(col):
                start_row = i*patch_size
                if start_row + patch_size > h:
                    start_row = h - patch_size 
                start_col = j*patch_size
                if start_col + patch_size > w:
                    start_col = w - patch_size
                
                res[index, :,:,:] = X[start_row:start_row+patch_size, start_col:start_col+patch_size, :]
                self.light_split_map.append([index, start_row, start_row+patch_size, start_col, start_col+patch_size])
                index += 1
        return res, resY
        
    def reconstruct_image_by_light_split(self, inputX, pathch_size=1):
        '''
        input shape is (batch, h, w, c)
        '''
        assert self.light_split_ori_shape is not None
        ori_h, ori_w, ori_c = self.light_split_ori_shape
        batch, h, w, c = inputX.shape
        assert batch == len(self.light_split_map) # light_split_map必须与batch值相同
        X = np.zeros((ori_h, ori_w, c))
        for tup in self.light_split_map:
            index, start_row, end_row, start_col, end_col = tup
            X[start_row:end_row, start_col:end_col, :] = inputX[index, :, :, :]
        return X


    def get_patches_by_split(self, X, Y, patch_size=1):
        h, w, c = X.shape
        row = h // patch_size
        col = w // patch_size
        newX = X
        res = np.zeros((row*col, patch_size, patch_size, c))
        resY = np.zeros((row*col))
        index = 0
        for i in range(row):
            for j in range(col):
                res[index,:,:,:] = newX[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size,:]
                index += 1
        self.split_row = row
        self.split_col = col
        return res, resY
    
    def split_to_big_image(self, splitX):
        '''
        input splitX shape (batch, 1, spe, h, w)
        return newX shape (spe, bigh, bigw)
        '''
        patch_size = self.patch_size
        batch, channel, spe, h, w = splitX.shape
        assert self.split_row * self.split_col == batch
        newX = np.zeros((spe, self.split_row * patch_size, self.split_col * patch_size))
        index = 0
        for i in range(self.split_row):
            for j in range(self.split_col):
                index = i * self.split_col + j
                newX[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = splitX[index, 0, :, :, :]
        return newX



    def re_build_split(self, X_patches, patch_size):
        '''
        X_pathes shape is (batch, channel=1, spectral, height, with)
        return shape is (height, width, spectral)
        '''
        h,w,c = self.data.shape
        row = h // patch_size
        if h % patch_size > 0:
            row += 1
        col = w // patch_size
        if  w % patch_size > 0:
            col += 1
        newX = np.zeros((c, row*patch_size, col*patch_size))
        for i in range(row):
            for j in range(col):
                newX[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = X_patches[i*col+j,0,:,:,:]
        return np.transpose(newX, (1,2,0))

    def get_patches(self, X, Y, patch_size=1, remove_zero=False):
        w,h,c = X.shape
        #1. padding
        margin = (patch_size - 1) // 2
        if self.padding:
            X_padding = self._padding(X, margin=margin)
        else:
            X_padding = X

        #2. zero patchs
        temp_w, temp_h, temp_c = X_padding.shape
        row = temp_w - patch_size + 1
        col = temp_h - patch_size + 1
        X_patchs = np.zeros((row * col, patch_size, patch_size, c)) #one pixel one patch with padding
        Y_patchs = np.zeros((row * col))
        patch_index = 0
        for r in range(0, row):
            for c in range(0, col):
                temp_patch = X_padding[r:r+patch_size, c:c+patch_size, :]
                X_patchs[patch_index, :, :, :] = temp_patch
                patch_index += 1

        if remove_zero:
            X_patchs = X_patchs[Y_patchs>0,:,:,:]
            Y_patchs = Y_patchs[Y_patchs>0]
            Y_patchs -= 1
        return X_patchs, Y_patchs #(batch, w, h, c), (batch)
    

    def custom_process(self, data):
        '''
        pavia数据集 增加一个光谱维度 从103->104 其中第104维为103的镜像维度
        data shape is [h, w, spe]
        '''

        if self.data_sign == "Pavia":
            h, w, spe = data.shape
            new_data = np.zeros((h,w,spe+1))
            new_data[:,:,:spe] = data
            new_data[:,:,spe] = data[:,:,spe-1]
            return new_data
        if self.data_sign == "Salinas":
            h, w, spe = data.shape
            new_data = np.zeros((h,w,spe+4))
            new_data[:,:,2:spe+2] = data
            return new_data
        return data


    def generate_torch_dataset(self, args, split=False, light_split=False):
        self.data, self.labels = self.load_data()

        data_hsi = self.data.reshape(
            np.prod(self.data.shape[:2]), np.prod(self.data.shape[2:]))
        pca = PCA(n_components=args['channel'])
        data_hsi = pca.fit_transform(data_hsi)
        data_hsi = preprocessing.scale(data_hsi)
        norm_data = data_hsi.reshape(self.data.shape[0], self.data.shape[1], args['channel'])

        
        X_patchs, Y_patchs = self.createImageCubes_even(norm_data, self.labels, windowSize=self.patch_size)

        X_all = X_patchs.transpose((0, 3, 1, 2))
        print('------[data] after transpose train, test------')
        print("X.shape=", X_all.shape)
        print("Y.shape=", Y_patchs.shape)

        trainset = TrainDS(X_all, Y_patchs)
        
        return trainset

       


if __name__ == "__main__":
    dataloader = HSIDataLoader(
        {"data":{"data_sign":"Houston", "padding":False, "batch_size":256, "patch_size":16, "select_spectral":[]}})
    train_loader,X,Y = dataloader.generate_torch_dataset(light_split=True)
    print(X.shape)
