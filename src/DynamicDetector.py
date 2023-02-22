import argparse
import os
from os.path import join, exists, isfile
import cv2
import sys
sys.path.append('/home/endeleze/Desktop/SECEDER/src/third_party')
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageTk, ImageDraw, ImageOps

import numpy as np
import pytorch_NetVlad.netvlad as netvlad
from SuperPoint_SuperGlue.base_model import dynamic_load
from SuperPoint_SuperGlue import extractors
from SuperPoint_SuperGlue import matchers

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=None, required=True,
                        help='path to data folder')
    parser.add_argument('--src_dir1', default=None, required=True,
                        help='path to src_dir 1')
    parser.add_argument('--src_dir0', default=None, required=True,
                        help='path to src_dir 0')
    parser.add_argument('--cpu', action='store_true', help='cpu for global descriptors')
    parser.add_argument('--ckpt_path', type=str, default='vgg16_netvlad_checkpoint',
                        help='Path to load checkpoint from, for resuming training or testing.')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='basenetwork to use', choices=['vgg16', 'alexnet'])
    parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
    parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
                        choices=['netvlad', 'max', 'avg'])
    parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
    parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
    parser.add_argument('--test_n', type=int, help='index of figure')
    opt = parser.parse_args()
    return opt

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class NetVladFeatureExtractor:
    def __init__(self, ckpt_path, arch='vgg16', num_clusters=64, pooling='netvlad', vladv2=False, nocuda=False,
                 input_transform=input_transform()):
        self.input_transform = input_transform

        flag_file = join(ckpt_path, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = json.load(f)
                stored_num_clusters = stored_flags.get('num_clusters')
                if stored_num_clusters is not None:
                    num_clusters = stored_num_clusters
                    print(f'restore num_clusters to : {num_clusters}')
                stored_pooling = stored_flags.get('pooling')
                if stored_pooling is not None:
                    pooling = stored_pooling
                    print(f'restore pooling to : {pooling}')

        cuda = not nocuda
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --nocuda")

        self.device = torch.device("cuda" if cuda else "cpu")

        print('===> Building model')

        if arch.lower() == 'alexnet':
            encoder_dim = 256
            encoder = models.alexnet(pretrained=True)
            # capture only features and remove last relu and maxpool
            layers = list(encoder.features.children())[:-2]

            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

        elif arch.lower() == 'vgg16':
            encoder_dim = 512
            encoder = models.vgg16(pretrained=True)
            # capture only feature part and remove last relu and maxpool
            layers = list(encoder.features.children())[:-2]

            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False

        encoder = nn.Sequential(*layers)
        self.model = nn.Module()
        self.model.add_module('encoder', encoder)

        if pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=vladv2)
            self.model.add_module('pool', net_vlad)
        else:
            raise ValueError('Unknown pooling type: ' + pooling)

        resume_ckpt = join(ckpt_path, 'checkpoints', 'checkpoint.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            best_metric = checkpoint['best_score']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model = self.model.eval().to(self.device)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    def feature(self, image):
        if self.input_transform:
            image = self.input_transform(image)
            # batch size 1
            image = torch.stack([image])

        with torch.no_grad():
            input = image.to(self.device)
            image_encoding = self.model.encoder(input)
            vlad_encoding = self.model.pool(image_encoding)
            del input
            torch.cuda.empty_cache()
            return vlad_encoding.detach().cpu().numpy()

class DynamicDetector():
    def __init__(self,opt):
        self.opt=opt
        self.test_n=opt.test_n
        self.src_dir1=opt.src_dir1
        self.src_dir0 = opt.src_dir0
        self.data_path=opt.data_path
        self.data_list=[os.path.join(self.data_path,i) for i in sorted(os.listdir(self.data_path))]
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.conf = {
            'output': 'feats-superpoint-n4096-r1600',
            'model': {
                'name': 'superpoint',
                'nms_radius': 4,
                'max_keypoints': 4096,
            },
            'preprocessing': {
                'grayscale': True,
                'resize_max': 1600,
            }}
        self.conf_match = {
            'output': 'matches-superglue',
            'model': {
                'name': 'superglue',
                'weights': 'outdoor',
                'sinkhorn_iterations': 50,
            },
        }
        Model_sp = dynamic_load(extractors, self.conf['model']['name'])
        self.sp_model = Model_sp(self.conf['model']).eval().to(self.device)
        Model_sg = dynamic_load(matchers, self.conf_match['model']['name'])
        self.sg_model = Model_sg(self.conf_match['model']).eval().to(self.device)
        self.extractor = NetVladFeatureExtractor(opt.ckpt_path, arch=opt.arch, num_clusters=opt.num_clusters,
                                                 pooling=opt.pooling, vladv2=opt.vladv2, nocuda=opt.nocuda)

    def prepare_data(self, image):
        image = np.array(ImageOps.grayscale(image)).astype(np.float32)
        image = image[None]
        data = torch.from_numpy(image / 255.).unsqueeze(0)
        return data

    def tensor_from_names(self, names, hfile):
        desc = [hfile[i]['global_descriptor'].__array__() for i in names]
        if self.opt.cpu:
            desc = torch.from_numpy(np.stack(desc, 0)).float()
        else:
            desc = torch.from_numpy(np.stack(desc, 0)).to(self.device).float()
        return desc

    def extract_local_features(self, image0):
        data0 = self.prepare_data(image0)
        pred0 = self.sp_model(data0.to(self.device))
        del data0
        torch.cuda.empty_cache()
        pred0 = {k: v[0].cpu().detach().numpy() for k, v in pred0.items()}
        if 'keypoints' in pred0:
            pred0['keypoints'] = (pred0['keypoints'] + .5) - .5
        pred0.update({'image_size': np.array([image0.size[0], image0.size[1]])})
        return pred0

    def geometric_verification(self, feats1, feats0):
        data = {}
        feats={}
        for k in feats0.keys():
            data[k + '0'] = feats0[k]
        for k in feats0.keys():
            data[k + '1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(self.device)
                for k, v in data.items()}
        data['image0'] = torch.empty((1, 1,) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1,) + tuple(feats1['image_size'])[::-1])
        pred = self.sg_model(data)
        matches = pred['matches0'][0].detach().cpu().short().numpy()
        pts0, pts1 ,lms= [], [],[]
        points0,scores0,descriptors0=[],[],[]
        for n, m in enumerate(matches):
            if m!=-1:
                pts0.append(feats0['keypoints'][n].tolist())
                scores0.append(feats0['scores'][n])
                points0.append(feats0['keypoints'][n])
                descriptors0.append(feats0['descriptors'][:,n])
                pts1.append(feats1['keypoints'][m].tolist())
        points0=np.array(points0)
        scores0=np.array(scores0)
        descriptors0=np.array(descriptors0).T
        feats={'keypoints':points0, 'scores':scores0, 'descriptors':descriptors0, 'image_size':feats0['image_size']}
        del data, feats1, pred


        try:
            pts0_ = np.int32(pts0)
            pts1_ = np.int32(pts1)
            F, mask = cv2.findFundamentalMat(pts0_, pts1_, cv2.RANSAC,ransacReprojThreshold = 1)
            # valid = sum(inliers)
            valid = len(pts0_[mask.ravel() == 1])
        except:
            valid=0
        pt_dynamic,pt_static=[],[]
        for ind,m in enumerate(mask):
            if m:
                pt_static.append(pts0[ind])
            else:
                pt_dynamic.append(pts0[ind])
        torch.cuda.empty_cache()
        pt0, pt1=pts0, pts1
        return pt_static,pt_dynamic,feats

    def dynamic_detector(self):
        pt_static,pt_dynamic=[],[]
        image0 = Image.open(self.data_list[self.test_n])
        width, height = image0.size
        scale = 640 / width
        newsize = (640, int(height * scale))
        image0 = image0.resize(newsize)
        # self.query_desc1 = self.extractor.feature(image1)[0]
        feats0 = self.extract_local_features(image0)
        data_min_ind=max(0,self.test_n-5)
        data_max_ind=min(len(self.data_list),self.test_n+5)
        for ind in range(data_min_ind,data_max_ind):
            if ind!=self.test_n:
                src_dir1=self.data_list[ind]
                image1 = Image.open(src_dir1)
                image1 = image1.resize(newsize)
                feats1 = self.extract_local_features(image1)
                pt_static,pt_dynamic,feats0 = self.geometric_verification(feats1, feats0)
        draw=ImageDraw.Draw(image0)
        for i in pt_static:
            x_,y_=i
            draw.ellipse((x_ - 2, y_ - 2, x_ + 2, y_ + 2), fill=(255, 0, 0))
        for i in pt_dynamic:
            x_,y_=i
            draw.ellipse((x_ - 2, y_ - 2, x_ + 2, y_ + 2), fill=(0, 255, 0))
        image0.show('f')
def main(opt):
    detector=DynamicDetector(opt)
    detector.dynamic_detector()

if __name__ == '__main__':
    opt = options()
    main(opt)
