import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

class selective_ts(nn.Module):
    def __init__(self, dim, r=2):
        super(selective_ts, self).__init__()
        self.dim = dim
        self.M = 19
        d = self.dim // r
        self.fc = nn.Linear(self.dim, d)
        self.fcs = nn.ModuleList([])
        for i in range(self.M):
            self.fcs.append(
                nn.Linear(d, self.dim)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        shape = list(x.size())
        if len(shape)==2:
            bs = shape[0]
        else:
            bs = 1
            x = x.unsqueeze(dim=0)
        x = x.view(bs,19,-1)
        
        fea_u = torch.sum(x,dim=1)
        fea_z = self.fc(fea_u)
        
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        fea_v = (x * attention_vectors).sum(dim=1)
        return fea_v

class selective_ts_with_g(nn.Module):
    def __init__(self, dim, r=2):
        super(selective_ts_with_g, self).__init__()
        self.dim = dim
        self.M = 19
        d = self.dim // r
        self.fc = nn.Sequential(
            nn.Linear(self.dim, d),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=d),
            nn.Linear(d, d))
        self.fcs = nn.ModuleList([])
        self.g_fcs = nn.ModuleList([])
        for i in range(self.M):
            self.fcs.append(
                nn.Linear(d, d)
            )
            self.g_fcs.append(
                nn.Linear(self.dim + d, self.dim)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, g):
        shape = list(x.size())
        if len(shape)==2:
            bs = shape[0]
        else:
            bs = 1
            x = x.unsqueeze(dim=0)
            g = g.unsqueeze(dim=0)
        x = x.view(bs,19,-1)
        g = g.view(bs,19,-1)
        fea_u = torch.sum(x,dim=1)
        fea_z = self.fc(fea_u)
        for i, (fc,gc) in enumerate(zip(self.fcs,self.g_fcs)):
            vector = fc(fea_z)
            vector = torch.cat([vector , g[:,i,:]], dim=1).unsqueeze(dim=1)
            vector = gc(vector)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        
        fea_v = (x * attention_vectors).sum(dim=1)
        return fea_v

class classifier_selective_ts_with_g(nn.Module):
    def __init__(self, numpy_class, dim):
        super(classifier_selective_ts_with_g, self).__init__()
        dim = 768
        self.scale = 64
        self.dy_att = selective_ts_with_g(dim)
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class)
            )
    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x, g): #[bs,dim]
        fea_dyn_select = self.dy_att(x,g)
        return self.layers(fea_dyn_select)

class classifier_selective_ts(nn.Module):
    def __init__(self, numpy_class, dim):
        super(classifier_selective_ts, self).__init__()
        dim = 768
        self.scale = 64
        self.dy_att = selective_ts(dim)
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class)
            )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x): #[bs,dim]
        fea_dyn_select = self.dy_att(x)
        return self.layers(fea_dyn_select)


def predict_labels(models, features, globs=None, size=1028, if_glob_features=False):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
        if if_glob_features: globs = torch.from_numpy(globs)
    
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            if if_glob_features:
                preds = models[MODEL_NUMBER](features.cuda(),globs.cuda())
            else:
                preds = models[MODEL_NUMBER](features.cuda())
            #preds = models[MODEL_NUMBER](features.cuda())
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            #img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        #js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        #top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 1)[0]
    return img_seg_final
    
def compute_HSI(args, preds, gts, dataset="indian_pines",print_per_class_ious=True):
    if dataset == "indian_pines":
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']
    elif dataset == "paviaU":
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']
    elif dataset == "houston2013":
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', ' Trees'
        , 'Soil', 'Water', 'Residential','Commercial', 'Road', 'Highway', 'Railway',
                    'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']
    else:
        target_names = ['Healthy grass', 'Stressed grass', 'Artificial turf', ' Evergreen trees'
        , 'Deciduous trees', 'Bare earth', 'Water','Residential buildings', 'Non-residential buildings', 'Roads', 'Sidewalks',
                    'Crosswalks', 'Major thoroughfares', 'Highways', 'Railways', 'Paved parking lots', 'Unpaved parking lots', 
                    'Cars', 'Trains', 'Stadium seats']
                
    ids = range(args['number_class'])
    size = args['image_size']
    pred_list = []
    label_list = []
    for pred, gt in zip(preds, gts):
        #predict = pred[size//2,size//2]
        #label = gt[size//2,size//2]
        pred_list.append(pred)
        label_list.append(gt)
    y_pred_test = np.array(pred_list)
    y_test = np.array(label_list)
    np.savetxt('results.txt',y_pred_test)
    #print(y_pred_test,y_test)
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names, labels=ids)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    
    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

def load_ensemble(args, device='cpu', if_glob_features=False):
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        if if_glob_features:
            model = nn.DataParallel(classifier_selective_ts_with_g(args["number_class"], args['dim'][-1]))
        else:
            model = nn.DataParallel(classifier_selective_ts(args["number_class"], args['dim'][-1]))
        model.load_state_dict(state_dict)
        model = model.module.to(device)
        models.append(model.eval())
    return models

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
