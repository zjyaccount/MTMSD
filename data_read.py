import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time, json
import os, sys
import copy

""" Training dataset"""

class HSIDS(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = torch.LongTensor(y)
        self.len = x.shape[0]

    def __getitem__(self, index):
        patch = torch.FloatTensor(self.x[index])
        return patch, self.y[index]
    
    def __len__(self):
        return self.len

class HSIDS_two(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = torch.LongTensor(y)
        self.len = x.shape[0]

    def __getitem__(self, index):
        central = torch.FloatTensor(self.x[index, 0, :])
        glob_feat = torch.FloatTensor(self.x[index, 1, :])
        return central, glob_feat, self.y[index]
    
    def __len__(self):
        return self.len        
    
class HSIFeatureDataLoader(object):
    def __init__(self, args, diffusion_data_path) -> None:
        self.data_path_prefix = './save_features'
        self.data = None #shape=(h,w,c)
        self.labels = None #shape=(h,w,1)
        self.seed = args['seed']
        # configs
        self.data_sign = args['dataset']
        self.patch_size = args['feature_patch_size']
        self.remove_zeros = True
        self.batch_size = args['batch_size']
        self.norm_type ='max_min' # 'none', 'max_min', 'mean_var'
        self.test_ratio = args['test_ratio']
        self.diffusion_data_path = diffusion_data_path

    def load_data_from_diffusion(self, data_ori, labels):
        path = "%s/%s" % (self.data_path_prefix, self.diffusion_data_path)
        data = np.load(path)
        patchesLabels = np.zeros((labels.shape[0] * labels.shape[1]),dtype="float16")
        patchIndex = 0
        for r in range(labels.shape[0]):
            for c in range(labels.shape[1]):
                patchesLabels[patchIndex] = labels[r, c]
                patchIndex = patchIndex + 1
        if data.shape[0] == patchesLabels.shape[0]: data = data[patchesLabels>0]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
        print("load diffusion data shape is ", data.shape)
        return data, patchesLabels

    def load_raw_data(self):
        data, labels = None, None
        if self.data_sign == "Indian_Pines":
            data = sio.loadmat('./datasets/indian_pines_16/Indian_pines_corrected.mat')['indian_pines_corrected']
            labels = sio.loadmat('./datasets/indian_pines_16/Indian_pines_gt.mat')['indian_pines_gt']
        elif self.data_sign == "PaviaU":
            data = sio.loadmat('./datasets/PaviaU_9/PaviaU.mat')['paviaU'] 
            labels = sio.loadmat('./datasets/PaviaU_9/PaviaU_gt.mat')['paviaU_gt']
        elif self.data_sign ==  "Houston2013":
            data = sio.loadmat('./datasets/houston_2013_15/Houston2013_corrected.mat')['input']
            labels = sio.loadmat('./datasets/houston_2013_15/Houston2013_gt.mat')['houston2013_gt']
        elif self.data_sign == "Houston2018":
            data = sio.loadmat('./datasets/Houston2018/Houston2018.mat')['houston2018']
            labels = sio.loadmat('./datasets/Houston2018/Houston2018_gt.mat')['houston2018_gt']
        elif self.data_sign == "WHU-Hi-LongKou":
            data = sio.loadmat('./datasets/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
            labels = sio.loadmat('./datasets/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
        else:
            pass
        h, w, _= data.shape
        return data, labels, h, w

    def load_data(self):
        ori_data, labels, h, w = self.load_raw_data()
        diffusion_data, diffusion_labels = self.load_data_from_diffusion(ori_data, labels)
        del ori_data, labels
        return diffusion_data, diffusion_labels, h, w
        
    def load_data_orign(self):
        ori_data, labels, h, w = self.load_raw_data()
        diffusion_data, diffusion_labels = self.load_data_from_diffusion(ori_data, labels)
        del ori_data
        labels = labels.reshape(labels.shape[0] * labels.shape[1])
        return diffusion_data, diffusion_labels, h, w, labels

    def _padding(self, X, margin=2):
        # pading with zeros
        w,h,c = X.shape
        new_x, new_h, new_c = w+margin*2, h+margin*2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x+w, start_y:start_y+h,:] = X
        return returnX
    
    def get_valid_num(self, y):
        tempy = y.reshape(-1)
        validy = tempy[tempy > 0]
        print('valid y shape is ', validy.shape)
        return validy.shape[0]
        
    def get_central_vectors(self, X, y):
        centralData = np.zeros((X.shape[0] * X.shape[1], 1, 1, X.shape[2]),dtype="float16")
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]),dtype="float16")
        patchIndex = 0
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                centralData[patchIndex, :, :, :] = X[r, c, :]
                patchesLabels[patchIndex] = y[r, c]
                patchIndex = patchIndex + 1
        centralData = centralData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
        return centralData, patchesLabels
    
    def get_central_glob_vectors(self, X, y):
        h, w, c = X.shape
        # x padding
        windowSize = self.patch_size
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = self._padding(X, margin=margin)
        whole_data = np.zeros((X.shape[0] * X.shape[1], 2, 1, 1, X.shape[2]),dtype="float16")
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]),dtype="float16")
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                glob_feat = np.mean(patch, axis=(0, 1))
                whole_data[patchIndex, 1, :, :, :] = glob_feat
                whole_data[patchIndex, 0, :, :, :] = zeroPaddedX[r, c, :]
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchIndex = patchIndex + 1

        whole_data = whole_data[patchesLabels>0,:,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
        return whole_data, patchesLabels

    def applyPCA(self, X, numComponents=30):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX

    def mean_var_norm(self, data):
        print("use mean_var norm...")
        h, w, c = data.shape
        data = data.reshape(h * w, c)
        data = StandardScaler().fit_transform(data)
        data = data.reshape(h, w, c)
        return data

    def cal_criterion(self, diff_feats, lamda=0.5, k=640):
        cate_num, timestep_num, feat_dim = diff_feats.shape  #label_num*19*dim

        original_feats = diff_feats.copy()
        for j in range(19):
            for i in range(feat_dim):
                diff_feats[:,j,i] = (diff_feats[:,j,i]-np.min(diff_feats[:,j,i]))/(np.max(diff_feats[:,j,i])-np.min(diff_feats[:,j,i]))
        print('Calculating criterion...')        
        sim_sum = np.zeros((19,feat_dim))
        for t in range(19):
            for i in range(cate_num):
                for j in range(cate_num):
                    if i != j:
                        sim_sum[t] += diff_feats[i, t, :] * diff_feats[j, t, :]
                        
        sim = sim_sum / (cate_num * (cate_num-1))
        criterion_class = np.zeros(feat_dim)
        for t in range(19):
            criterion_class += ((-1) * lamda * sim[t] + (1-lamda) * np.var(diff_feats[:,t,:], axis=0))
        
        diff_feats = original_feats
        for j in range(cate_num):
            for i in range(feat_dim):
                diff_feats[j,:,i] = (diff_feats[j,:,i]-np.min(diff_feats[j,:,i]))/(np.max(diff_feats[j,:,i])-np.min(diff_feats[j,:,i]))
        sim_sum = np.zeros((cate_num,feat_dim))
        for cate in range(cate_num):
            for i in range(19):
                for j in range(19):
                    if i != j:
                        sim_sum[cate] += diff_feats[cate, i, :] * diff_feats[cate, j, :]
                        
        sim = sim_sum / (19*(19-1))
        criterion_t = np.zeros(feat_dim)
        for cate in range(cate_num):
            criterion_t += ((-1) * lamda * sim[cate] + (1-lamda) * np.var(diff_feats[cate,:,:], axis=0))
        
        criterion = criterion_class + 0.5*criterion_t
        criterion = torch.from_numpy(criterion)
        _, indices = torch.topk(criterion, k=k)
        indices = indices.numpy()
        return indices

    def data_preprocessing(self, data):
        '''
        1. normalization
        2. pca
        3. spectral filter
        data: [h, w, spectral]
        '''
        if self.norm_type == 'max_min':
            norm_data = np.zeros(data.shape)
            for i in range(data.shape[2]):
                input_max = np.max(data[:,:,i])
                input_min = np.min(data[:,:,i])
                norm_data[:,:,i] = (data[:,:,i]-input_min)/(input_max-input_min)
        elif self.norm_type == 'mean_var':
            norm_data = self.mean_var_norm(data)
        else:
            norm_data = data 
        pca_num = self.data_param.get('pca', 0)
        if pca_num > 0:
            print('before pca')
            pca_data = self.applyPCA(norm_data, int(self.data_param['pca']))
            norm_data = pca_data
            print('after pca')
        return norm_data
        
    def prepare_data(self, if_glob_features=True):        
        self.data, self.labels, h, w = self.load_data()
        
        x_train, x_test, y_train, y_test = train_test_split(self.data,
                                                        self.labels,
                                                        test_size=self.test_ratio,
                                                        random_state=self.seed,
                                                        stratify=self.labels)
        
        print('[load data done.] load data shape data=%s, label=%s' % (str(self.data.shape), str(self.labels.shape)))
        
        if self.data_sign == "Indian_Pines":
            label_num = 16
        elif self.data_sign == "PaviaU":
            label_num = 9
        elif self.data_sign == "Houston2018":
            label_num = 20
        elif self.data_sign == "WHU-Hi-LongKou":
            label_num = 9
        
        #train sample of each class >0
        for i in range(label_num):
            if i not in y_train:
                index = np.where(y_test==i)[0][0]
                print(index)
                print(i)
                x_train = np.insert(x_train,-1,x_test[index],axis=0)
                y_train = np.insert(y_train,-1,y_test[index])
                x_test = np.delete(x_test,index,0)
                y_test = np.delete(y_test,index,0)
        
        #Purification
        avelabel_x_train = np.zeros((label_num, 19, x_train.shape[2]//19))
        avelabel_g_train = np.zeros((label_num, 19, x_train.shape[2]//19))
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 19, x_train.shape[2]//19) #[bs,2,19,dim]
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 19, x_test.shape[2]//19)

        for i in range(label_num):
            for j in range(19):
                x_labels = x_train[y_train == i, 0, j, :]
                avelabel_x_train[i, j, :] = np.mean(x_labels, axis=0)

        
        for i in range(label_num):
            for j in range(19):
                g_labels = x_train[y_train == i, 1, j, :]
                avelabel_g_train[i, j, :] = np.mean(g_labels, axis=0)
        k = 768
        indices_x = self.cal_criterion_same(avelabel_x_train, lamda=0.8, k=k)
        indices_g = self.cal_criterion_same(avelabel_g_train, lamda=0.8, k=k)
        x_train[:, 0, :, :k] = x_train[:, 0, :, :][:,:,indices_x]  
        x_test[:, 0, :, :k] = x_test[:, 0, :, :][:,:,indices_x]
        x_train[:, 1, :, :k] = x_train[:, 1, :, :][:,:,indices_g]
        x_test[:, 1, :, :k] = x_test[:, 1, :, :][:,:,indices_g]
        x_train = np.delete(x_train,np.s_[k:],axis = 3)
        x_test = np.delete(x_test,np.s_[k:],axis = 3)
        
        self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 19, self.data.shape[2]//19)
        self.data[:, 0, :, :k] = self.data[:, 0, :, :][:,:,indices_x]
        self.data[:, 1, :, :k] = self.data[:, 1, :, :][:,:,indices_g]
        self.data = np.delete(self.data,np.s_[k:],axis = 3)
        self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], self.data.shape[2]*self.data.shape[3])
        
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2] * x_train.shape[3])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2] * x_test.shape[3])
        

        if if_glob_features:
            trainset = HSIDS_two(x_train, y_train) 
            testset = HSIDS_two(x_test, y_test)  
            allset = HSIDS_two(self.data, self.labels)
        else:
            x_train = x_train[:, 0, :]
            x_test = x_test[:, 0, :]
            trainset = HSIDS(x_train, y_train) 
            testset = HSIDS(x_test, y_test)  
            allset = HSIDS(self.data, self.labels)

        print('------[data] split data to train, test------')
        print("train len: %s" % len(trainset))
        print("test len : %s" % len(testset))
        print("all len: %s" % len(allset))
        return trainset, testset, allset, h, w, self.labels
 
    def generate_torch_dataset(self, if_glob_features=False):
        trainset, testset, allset,h,w, labels = self.prepare_data(if_glob_features)
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                drop_last=True
                                                )

        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                batch_size=128,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False
                                                )
        all_loader = torch.utils.data.DataLoader(dataset=allset,
                                                batch_size=128,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False
                                                )
        return train_loader, all_loader, all_loader, h, w, labels

def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]
       


if __name__ == "__main__":
    dataloader = HSIFeatureDataLoader({"data":{"data_path_prefix":'../../data', "data_sign": "Indian",
        "data_file": "Indian_40"}})
    train_loader, unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset()