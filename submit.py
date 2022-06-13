import warnings
warnings.filterwarnings("ignore")


from distutils.command.build import build
from config import *
#from eval_util import  eval_dist
from utils.logger import init_logger_eval
from util import build_model, build_dataloader
from utils.util import seed_everything, torch_distributed_zero_first

import pandas as pd
import numpy as np
import sys
import os
import time
import cv2
import random
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


import albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2



from torch_scatter import scatter   

import torch.distributed as dist

from net import *


class SnakeNetTeacher(nn.Module):
	def __init__(self, cfg, margins=None):
		super(SnakeNetTeacher, self).__init__()

		self.backbone = BackBone(cfg)
		self.cfg = cfg    


		self.embedding_size = cfg.embedding_size

		self.global_pool = nn.AdaptiveAvgPool2d(1)


		self.out_features = self.backbone.out_features


		if cfg.use_meta:
			self.ArcHead_id_meta = ArcMarginProduct_subcenter(self.backbone.out_features * 2, cfg.num_classes)
		else:
			self.ArcHead_id = ArcMarginProduct_subcenter(self.backbone.out_features, cfg.num_classes)
		
		self.bn = nn.BatchNorm1d(self.backbone.out_features)


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


	#@autocast()
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

  
		return cosine_id, None





# FungDatasetTest
class SnakeDatasetTest(Dataset):
	def __init__(self, cfg):
		self.cfg = cfg
		
		self.root = cfg.root
		root = self.root
		
		self.train_df = pd.read_csv(f'{root}/SnakeCLEF2022-TrainMetadata.csv')
		self.df = pd.read_csv(f'{root}/SnakeCLEF2022-TestMetadata.csv')
		self.image_dir = f'{root}/SnakeCLEF2022-test_images/SnakeCLEF2022-large_size'


		self.df.loc[self.df.code == 'KM', 'code'] = 'unknown'
		self.df.loc[~self.df.country.isin(self.train_df.country.unique()), 'country'] = 'unknown'

		self.df_indexs = self.df.index.tolist()


		cfg.logger.info(f'测试集数量: {len(self.df_indexs)}')

		self.code2label = {}
		self.country2label = {}

		with open(f'{cfg.root}/code.txt', 'r') as f:
			for line in f.readlines():
				#print(line.strip().split(','))
				name, label = line.strip().split(':')
				label = int(label)
				self.code2label[name] = label

		with open(f'{cfg.root}/country.txt', 'r') as f:
			for line in f.readlines():
				#print(line.strip().split(','))
				name, label = line.strip().split(':')
				label = int(label)
				self.country2label[name] = label


		self.transforms = A.Compose([
			A.Resize(cfg.image_size, cfg.image_size),
		], p=1.0)



		self.mean = np.array([0.485, 0.456, 0.406])
		self.std = np.array([0.229, 0.224, 0.225])        
		
		


	def __getitem__(self, idx):
		
		index = self.df_indexs[idx]
		item = self.load_data(index)

		return item
   

	def load_transform_data(self, img, transform):
		cfg = self.cfg
		h, w, c = img.shape

		image = transform(image=img)['image']
		image = (image / 255. - self.mean) / self.std

		image = self.img2tensor(image)

		return image		
		

	def load_TTA_images(self, img):
		cfg = self.cfg
		h, w, c = img.shape

		transforms_0 = A.Compose([
			A.Resize(cfg.image_size, cfg.image_size, always_apply=True, p=1),
			A.HorizontalFlip(always_apply=True, p=1),
		])

		transforms_1 = A.Compose([
			A.Resize(cfg.image_size, cfg.image_size, always_apply=True, p=1),
			A.VerticalFlip(always_apply=True, p=1),
		])


		transforms_2 = A.Compose([
			A.CenterCrop(height=int(h * 0.85), width=int(w * 0.85), always_apply=True, p=1),
			A.Resize(cfg.image_size, cfg.image_size, always_apply=True, p=1),
		])

		transforms_3 = A.Compose([
			A.CenterCrop(height=int(h * 0.75), width=int(w * 0.75), always_apply=True, p=1),
			A.Resize(cfg.image_size, cfg.image_size, always_apply=True, p=1),
		])


		transforms_4 = A.Compose([
			A.CenterCrop(height=int(h * 0.8), width=int(w * 0.8), always_apply=True, p=1),
			A.Resize(cfg.image_size, cfg.image_size, always_apply=True, p=1),
		])

		transforms_5 = A.Compose([
			A.CenterCrop(height=int(h * 0.7), width=int(w * 0.7), always_apply=True, p=1),
			A.Resize(cfg.image_size, cfg.image_size, always_apply=True, p=1),
		])


		transforms_6 = A.Compose([
			A.CenterCrop(height=int(h * 0.9), width=int(w * 0.9), always_apply=True, p=1),
			A.Resize(cfg.image_size, cfg.image_size, always_apply=True, p=1),
			A.HorizontalFlip(always_apply=True, p=1),
		])


		transforms_7 = A.Compose([
			A.CenterCrop(height=int(h * 0.8), width=int(w * 0.8), always_apply=True, p=1),
			A.Resize(cfg.image_size, cfg.image_size, always_apply=True, p=1),
			A.HorizontalFlip(always_apply=True, p=1),
		])



		image_0 = self.load_transform_data(img, transforms_0)
		image_1 = self.load_transform_data(img, transforms_1)
		image_2 = self.load_transform_data(img, transforms_2)
		image_3 = self.load_transform_data(img, transforms_3)
		image_4 = self.load_transform_data(img, transforms_4)
		image_5 = self.load_transform_data(img, transforms_5)
		image_6 = self.load_transform_data(img, transforms_6)
		image_7 = self.load_transform_data(img, transforms_7)

		TTA_images = [image_0, image_1, image_2, image_3, image_4, image_5, image_6, image_7]

		return TTA_images




	def load_data(self, index):
		info = self.df.iloc[index]

		image_name = info['file_path']
		image_path = os.path.join(self.image_dir, image_name)

		observation_id = info['observation_id']
		endemic = info['endemic']
		country = info['country']
		code = info['code']
		code = int(self.code2label[code])
		country = int(self.country2label[country])
		endemic = 1 if endemic else 0
  
		meta = {}
		meta['endemic'] = endemic
		meta['code'] = code
		meta['country'] = country

		
		

		img = cv2.imread(image_path)

  
		image = self.transforms(image=img)['image']
		image = (image / 255. - self.mean) / self.std

		image = self.img2tensor(image)


		TTA_images = 1
		if self.cfg.use_TTA:
			TTA_images = self.load_TTA_images(img)


		item = {}
		item['image'] = image
		item['df_index'] = index
		item['observation_id'] = observation_id

		item['TTA_images'] = TTA_images

		item['meta'] = meta

		return item


	def img2tensor(self, img):
		img = np.transpose(img, (2, 0, 1)).astype(np.float32)
		img = torch.from_numpy(img)
		return img		


	def __len__(self):
		#return 1000
		return len(self.df_indexs)




def test_single(model, data_loader):
	all_image_names = []
	all_observation_ids = []
	all_probs = []

	model.eval()
	with torch.no_grad():
		for batch, data in enumerate(tqdm(data_loader)):
			images, meta, image_names = data['image'], data['meta'], data['df_index']
			observation_ids = data['observation_id'].numpy().tolist()
			images = images.cuda()
			
			TTA_images = data['TTA_images']

			for key in meta.keys():
				meta[key] = meta[key].cuda()
				
			probs = model(images.float(), meta=meta)[0]


			if cfg.use_TTA:
				for tta_image in TTA_images:
					tta_image = tta_image.cuda()
					tta_probs = model(tta_image.float(), meta=meta)[0]
					probs += tta_probs
				probs = probs / (len(TTA_images) + 1.)


			all_probs.append(probs.cpu().detach())


			all_image_names.extend(image_names)
			all_observation_ids.extend(observation_ids)

		all_probs = torch.cat(all_probs, dim=0)
		all_image_names = torch.tensor(all_image_names)
		all_observation_ids = torch.tensor(all_observation_ids)



	return all_image_names, all_observation_ids, all_probs






def test_dist(model, data_loader, cfg):
	time_start = time.time()
	world_size = dist.get_world_size()
	all_image_names_, all_observation_ids_, all_probs_ = test_single(model, data_loader)

	all_image_names, all_observation_ids, all_probs = [[None] * world_size for i in range(3)]


	dist.all_gather_object(all_image_names, all_image_names_)
	dist.all_gather_object(all_observation_ids, all_observation_ids_)
	dist.all_gather_object(all_probs, all_probs_)


	all_image_names = torch.cat(all_image_names, dim=0)
	all_observation_ids = torch.cat(all_observation_ids, dim=0)
	all_probs = torch.cat(all_probs, dim=0)


	#### 去除dist补全的重复数据
	N = all_probs.shape[0]

	keep = np.array([False] * N)
	all_image_names = all_image_names.numpy()

	keep[np.unique(all_image_names, return_index=True)[1]] = True	


	all_probs = all_probs[keep, :]
	all_observation_ids = all_observation_ids[keep]
	all_image_names = all_image_names[keep]


	#### index 排序
	# sort
	sort_df_index = np.argsort(all_image_names)	
	all_probs = all_probs[sort_df_index, :]
	all_observation_ids = all_observation_ids[sort_df_index]	

	np.save(f'submits/features/{cfg.backbone}_probs.npy', all_probs)
	np.save(f'submits/features/{cfg.backbone}_observation_ids.npy', all_observation_ids)



	#### 合并同一个id的
	N = all_probs.shape[0]

	# sort
	sort_index = np.argsort(all_observation_ids)

	all_probs = all_probs[sort_index, :]
	all_observation_ids = all_observation_ids[sort_index]

	# scatter
	unique_index = np.unique(all_observation_ids, return_index=True)[1]

	inds = np.zeros(shape=N, dtype=np.int64)	

	ind = 0
	for i in range(len(unique_index)-1):
		start = unique_index[i]
		num = unique_index[i+1] - start
		inds[start:start+num] = ind
		ind += 1
	inds[unique_index[-1]:] = ind	

	unique_probs = scatter(torch.tensor(all_probs), torch.tensor(inds), dim=0, reduce='mean')
	unique_observation_ids = all_observation_ids[unique_index]

	unique_pred_probs, unique_pred_labels = torch.max(unique_probs, 1)
	unique_pred_labels = unique_pred_labels.numpy()

	N = unique_pred_labels.shape[0]

	unique_observation_ids = unique_observation_ids.reshape(N, 1)
	unique_pred_labels = unique_pred_labels.reshape(N, 1)
	unique_pred_probs = unique_pred_probs.reshape(N, 1)
	data_array = np.concatenate([unique_observation_ids, unique_pred_labels, unique_pred_probs], axis=1)


	
	column_names_prob = ['ObservationId', 'class_id', 'prob']

	submit_df = pd.DataFrame(data_array, columns=column_names_prob)
	submit_df['ObservationId'] = submit_df['ObservationId'].astype(dtype='int')
	submit_df['class_id'] = submit_df['class_id'].astype(dtype='int')

	submit_df.to_csv(f'submits/submit_{cfg.backbone}_prob.csv', index=None)

	del submit_df['prob']
	submit_df.to_csv(f'submits/submit_{cfg.backbone}.csv', index=None)


	time_end = time.time()
	eval_time = time_end - time_start
	cfg.logger.info(f'test time: {eval_time:.0f}s')





def test(args, cfg):
    	
	cfg.samples_per_gpu = 16
	cfg.workers_per_gpu = 8
	cfg.use_TTA = True

	#### build data_loader
	dataset = SnakeDatasetTest(cfg)
	sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=False)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.samples_per_gpu, sampler=sampler, pin_memory=False, num_workers=cfg.workers_per_gpu)

	#### build model
	model = SnakeNetTeacher(cfg)


	#model_path = ''
	model_path = args.checkpoint_path
	model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
	model = model.cuda()


	# sync
	if args.local_rank != -1:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

	# 分布式训练 DistributedDataParallel
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
	

	test_dist(model, dataloader, cfg)






if __name__ == '__main__':
	
	seed_everything(42)
	cv2.setNumThreads(0)

	parser = argparse.ArgumentParser()	
	parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
	parser.add_argument('--name', default='mmcls123', type=str)
	parser.add_argument('--checkpoint_path', default='', type=str)
	parser.add_argument('--image_size', default=736, type=int)
	parser.add_argument('--backbone', default='', type=str)
	parser.add_argument('--use_meta', action='store_true')

	args = parser.parse_args()	

	print('local_rank: ', args.local_rank)

	if args.local_rank != -1:
		dist.init_process_group(backend='nccl')
		torch.cuda.set_device(args.local_rank)
	
	if args.local_rank == -1:
		cfg = Cfg()

		cfg.image_size = args.image_size
		cfg.backbone = args.backbone
		cfg.use_meta = args.use_meta

		print('init logger')
		cfg.work_dir, cfg.logger = init_logger_eval()
		print_cfg(cfg.logger, cfg)
	else:
		with torch_distributed_zero_first(args.local_rank):
			cfg = Cfg()

			cfg.image_size = args.image_size
			cfg.backbone = args.backbone
			cfg.use_meta = args.use_meta

			print('init logger')
			
			cfg.work_dir, cfg.logger = init_logger_eval()
			if args.local_rank == 0:
				print_cfg(cfg.logger, cfg)

	test(args, cfg)























