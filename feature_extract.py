from data_read import HSIDataLoader
from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from src.utils import setup_seed, multi_acc
import argparse
import json
import torch
import os
from tqdm import tqdm
import numpy as np
from guided_diffusion.guided_diffusion.dist_util import dev
from src.feature_extractors import create_feature_extractor, collect_features, collect_features_all

def extract(args, batch_size=16, patch_size=48, select_spectral=[]):
    res_feature = []
    res_glob_feature = []
    dataloader = HSIDataLoader({"data":{"data_sign":args['dataset'], "padding":False, "batch_size":batch_size, "patch_size":patch_size, "select_spectral":select_spectral}})
    all_set = dataloader.generate_torch_dataset(args)
    feature_extractor = create_feature_extractor(**args)
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, args['channel'], args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 
    for row, (img, _) in enumerate(tqdm(all_set)):
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        feas, glob_feats = collect_features(args, features)
        feas = feas.cpu().numpy()
        glob_feats = glob_feats.cpu().numpy()
        res_feature.append(feas)
        res_glob_feature.append(glob_feats)
    res_feature = np.expand_dims(np.stack(res_feature, axis=0), axis=1)
    res_glob_feature = np.expand_dims(np.stack(res_glob_feature, axis=0), axis=1)
    res_whole_feature = np.concatenate([res_feature, res_glob_feature], axis=1)
    print(f'Feature shape: {res_whole_feature.shape}')
    path = './save_features/%s/full_feature_central_globs.npy' % args['dataset']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, res_whole_feature)
    print(f'Feature saved to {path}')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--dataset', type=str, default="Houston2018") #加了数据集参数
    args = parser.parse_args()
    setup_seed(args.seed)
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]
    extract(opts, patch_size=opts['image_size'])
