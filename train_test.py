import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import gc
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import argparse
from src.utils import setup_seed, multi_acc
from src.classifier import load_ensemble, predict_labels, compute_HSI, classifier_selective_ts, classifier_selective_ts_with_g
from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev
import datetime
from data_read import HSIFeatureDataLoader

def evaluation(args, models, test_dataloader, if_glob_features=False):
    preds, gts, uncertainty_scores = [], [], []
    if if_glob_features:
        for img, g, label in tqdm(test_dataloader):        
            img, g = img.to(dev()), g.to(dev())
            pred = predict_labels(
                models, img, globs=g, size=args['dim'][:-1], if_glob_features=if_glob_features
            )
            gts.extend(list(label.numpy()))
            preds.extend(list(pred.numpy()))
    else:
        for img, label in tqdm(test_dataloader):        
            img = img.to(dev())
            pred = predict_labels(
                models, img, size=args['dim'][:-1], if_glob_features=if_glob_features
            )
            gts.extend(list(label.numpy()))
            preds.extend(list(pred.numpy()))
    classification, oa, confusion, each_acc, aa, kappa = compute_HSI(args, preds, gts)
    print(f'classification: ', classification)
    print(f'each acc:', each_acc)
    print(f'confusion:', confusion)
    print('oa: %.6f, aa: %.6f, kappa: %.6f: ' % ( oa, aa, kappa))


def train(args, train_dataloader, if_glob_features=False):

    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):
        gc.collect()
        if if_glob_features:
            classifier = classifier_selective_ts_with_g(numpy_class=(args['number_class']), dim=args['dim'][-1])
        else:
            classifier = classifier_selective_ts(numpy_class=(args['number_class']), dim=args['dim'][-1])
        classifier.init_weights()
        learning_rate = 0.001
        epochs = 100
        classifier = nn.DataParallel(classifier).cuda()
        criterion = nn.CrossEntropyLoss()
        print('lr: %.6f, epochs:%d, schedule:' % (learning_rate, epochs))
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
        classifier.train()

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        eta_min = 0.000005
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min)
        if if_glob_features:
            for epoch in range(epochs):
                for X_batch, G_batch, y_batch in train_dataloader:
                    X_batch, G_batch, y_batch = X_batch.to(dev()), G_batch.to(dev()), y_batch.to(dev())
                    y_batch = y_batch.type(torch.long)
                    optimizer.zero_grad()
                    y_pred = classifier(X_batch,G_batch)
                    loss = criterion(y_pred, y_batch)
                    acc = multi_acc(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

                    iteration += 1
                    if iteration % 1000 == 0:
                        print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                    
                    if epoch > 10:
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            break_count = 0
                        else:
                            break_count += 1

                        if break_count > 800:
                            stop_sign = 1
                            print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                            break
                scheduler.step()
                if stop_sign == 1:
                    break
        else:
            for epoch in range(epochs):
                for X_batch, y_batch in train_dataloader:
                    X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
                    y_batch = y_batch.type(torch.long)
                    optimizer.zero_grad()
                    y_pred = classifier(X_batch)
                    loss = criterion(y_pred, y_batch)
                    acc = multi_acc(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

                    iteration += 1
                    if iteration % 1000 == 0:
                        print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                    
                    if epoch > 10:
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            break_count = 0
                        else:
                            break_count += 1

                        if break_count > 800:
                            stop_sign = 1
                            print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                            break
                scheduler.step()
                if stop_sign == 1:
                    break
        model_path = os.path.join(args['exp_dir'], 
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    parser.add_argument('--exp', type=str)
    parser.add_argument('--test_ratio', type=float,  default=0)
    parser.add_argument('--dataset', type=str, default="Houston2013") #加了数据集参数
    parser.add_argument('--seed', type=int,  default=345)
    parser.add_argument('--feature_patch_size', type=int, default=3)
    args = parser.parse_args()
    setup_seed(args.seed)
    curr_time = datetime.datetime.now()
    time=datetime.datetime.strftime(curr_time,'%Y-%m-%d-%H-%M-%S')
    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    # Prepare the model folder 
    if len(opts['steps']) > 0:
        opts['exp_dir'] = os.path.join(opts['exp_dir'], time)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    if opts['dataset'] == 'Indian_Pines':
        testRatio = 0.9
    elif opts['dataset'] == 'WHU-Hi-LongKou':
        testRatio = 0.995
    elif opts['dataset'] == 'Houston2018':
        testRatio = 0.95
    elif opts['dataset'] == 'PaviaU':
        testRatio = 0.95
    else: print('dataset not found')
    
    if testRatio != 0: testRatio = args.test_ratio
    
    print(testRatio)
    
    if_glob_features = True
    
    diffusion_data_path = '/%s/full_feature_central_globs.npy' % opts['dataset']
    feature_dataloader = HSIFeatureDataLoader(opts, diffusion_data_path)
    train_dataloader, test_dataloader, _, _, _, _ = feature_dataloader.generate_torch_dataset(if_glob_features=if_glob_features)

    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
        
    if not all(pretrained):
        opts['start_model_num'] = sum(pretrained)
        train(opts, train_dataloader, if_glob_features=if_glob_features)
    
    print('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda',if_glob_features=if_glob_features) 
    evaluation(opts, models, test_dataloader, if_glob_features=if_glob_features) 
