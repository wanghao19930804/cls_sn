from torch.cuda import amp
import torchvision
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import copy
import timm


import math
import numpy as np


class BackBone(nn.Module):
	def __init__(self, cfg, pretrained=True):
		super(BackBone, self).__init__()
		self.net = timm.create_model(cfg.backbone, pretrained=pretrained)

		if 'efficientnet' in cfg.backbone:
			self.out_features = self.net.classifier.in_features
		elif 'tresnet' in cfg.backbone:
			self.out_features = self.net.head.fc.in_features
		elif 'res' in cfg.backbone: #works also for resnest
			self.out_features = self.net.num_features
		elif 'patch' in cfg.backbone or 'MetaFG' in cfg.backbone:
			self.out_features = self.net.head.in_features
		elif 'mixer' in cfg.backbone or 'convnext' in cfg.backbone or 'nfnet' in cfg.backbone:
			self.out_features = self.net.head.fc.in_features
		else:
			self.out_features = self.net.classifier.in_features

	def forward(self, x):
		x = self.net.forward_features(x)
		return x

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()

class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, n_classes, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
        self.out_dim =n_classes
            
    def forward(self, logits, labels):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss     


class FGVCNet(nn.Module):
	def __init__(self, cfg, margins=None):
		super(FGVCNet, self).__init__()

		self.backbone = BackBone(cfg)
		self.cfg = cfg    
  
		if cfg.fix_backbone:
			for p in self.backbone.parameters():
				p.requires_grad = False

		self.embedding_size = cfg.embedding_size


		self.global_pool = nn.AdaptiveAvgPool2d(1)

		self.out_features = self.backbone.out_features



		if cfg.use_meta:
			self.ArcHead_id_meta = ArcMarginProduct_subcenter(self.backbone.out_features * 2, cfg.num_classes)
		else:
			self.ArcHead_id = ArcMarginProduct_subcenter(self.backbone.out_features, cfg.num_classes)
		
		self.bn = nn.BatchNorm1d(self.backbone.out_features)
		self.loss_func_arc = ArcFaceLossAdaptiveMargin(margins, cfg.num_classes)



		if cfg.use_meta:
			# meta
			self.endemic_embedding = nn.Embedding(2, self.embedding_size)
			self.code_embedding = nn.Embedding(207, self.embedding_size)
			self.country_embedding = nn.Embedding(2152, self.embedding_size)

			self.neck_meta = nn.Sequential(
				nn.Linear(self.embedding_size * 3, self.embedding_size, bias=True),
				nn.BatchNorm1d(self.embedding_size),
				torch.nn.PReLU(),
				nn.Linear(self.embedding_size, self.out_features, bias=True),
				nn.BatchNorm1d(self.out_features),
			)


	def forward(self, x, labels=None, meta=None, lam=None, mix_indices=None, extract_feature=False):

		#### image
		x = self.backbone(x)

		feature = self.global_pool(x)[:, :, 0, 0] if len(x.shape) == 4 else x


		#### meta
		if self.cfg.use_meta:
			endemic = meta['endemic']
			code = meta['code']
			country = meta['country']

			endemic_feature = self.endemic_embedding(endemic)
			code_feature = self.code_embedding(code)
			country_feature = self.country_embedding(country)

			if lam is not None:
				endemic_feature = lam * endemic_feature + (1 - lam) * endemic_feature[mix_indices]
				code_feature = lam * code_feature + (1 - lam) * code_feature[mix_indices]
				country_feature = lam * country_feature + (1 - lam) * country_feature[mix_indices]

			meta_feature = torch.cat([endemic_feature, code_feature, country_feature], dim=-1)
			feature_meta = self.neck_meta(meta_feature)
	



		feature = self.bn(feature)


		if self.cfg.use_meta:
			feature = torch.cat([feature, feature_meta], dim=-1)
			cosine_id = self.ArcHead_id_meta(feature)
		else:
			cosine_id = self.ArcHead_id(feature)

		if extract_feature:
			return feature			




		#### loss
		loss = None
		arc_loss = 0
		fc_loss = 0
	
		if self.training:
			if lam is None:
				arc_loss = self.loss_func_arc(cosine_id, labels)
			else:
				arc_loss = lam * self.loss_func_arc(cosine_id, labels) + (1 - lam) * self.loss_func_arc(cosine_id, labels[mix_indices])
		
			loss = fc_loss + arc_loss

  
		return cosine_id, loss











