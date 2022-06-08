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

import copy


class SnakeDataset(Dataset):
	def __init__(self, cfg_, is_training):
		cfg = copy.deepcopy(cfg_)
		self.cfg = cfg
		self.is_training = is_training

		
		self.root = cfg.root
		root = self.root
		

		self.train_df = pd.read_csv(f'{root}/train_val_v3.csv')
		self.test_df = pd.read_csv(f'{root}/SnakeCLEF2022-TestMetadata.csv')
		self.image_dir = f'{root}/SnakeCLEF2022-large_size'


		if self.is_training:
			if cfg.use_all_data:
				self.df = self.train_df	
			else:
				self.df = self.train_df[self.train_df.fold_0 == 'train'].reset_index(drop=True)
		else:
			self.df = self.train_df[self.train_df.fold_0 == 'val'].reset_index(drop=True)
		


		self.df.loc[~self.df.code.isin(self.test_df.code.unique()), 'code'] = 'unknown'
		self.df.loc[~self.df.country.isin(self.test_df.country.unique()), 'country'] = 'unknown'

		tmp = np.sqrt(1 / np.sqrt(self.df.class_id.value_counts().sort_index().values))
		self.margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * cfg.arcface_m_x + cfg.arcface_m_y

		self.df_indexs = self.df.index.tolist()


		self.code2label = {}
		self.country2label = {}

		with open(f'{cfg.root}/code.txt', 'r') as f:
			for line in f.readlines():
				name, label = line.strip().split(':')
				label = int(label)
				self.code2label[name] = label

		with open(f'{cfg.root}/country.txt', 'r') as f:
			for line in f.readlines():
				name, label = line.strip().split(':')
				label = int(label)
				self.country2label[name] = label


		if self.is_training:
			cfg.logger.info(f'训练集数量: {len(self.df_indexs)}')
		else:
			cfg.logger.info(f'验证集数量: {len(self.df_indexs)}')





		self.train_transforms_list = [
			A.RandomResizedCrop(cfg.image_size, cfg.image_size, p=0.8),
			A.Resize(cfg.image_size, cfg.image_size),
			
			A.HorizontalFlip(p=0.5),
   			A.VerticalFlip(p=0.5),

			A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.6),
			A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
			A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),

			A.OneOf([
				A.OpticalDistortion(p=0.3),
				A.GridDistortion(p=0.1),
				A.IAAPiecewiseAffine(p=0.3),
			], p=0.2),   			

			A.Cutout(p=0.2),
		]

		
		self.train_transforms = A.Compose(self.train_transforms_list, p=1.0)

		self.val_transforms = A.Compose([
			A.Resize(cfg.image_size, cfg.image_size),
		], p=1.0)


		if self.is_training:
			self.transforms = self.train_transforms
		else:
			self.transforms = self.val_transforms


		self.mean = np.array([0.485, 0.456, 0.406])
		self.std = np.array([0.229, 0.224, 0.225])        
		
		


	def load_transform_data(self, img, transform):
		cfg = self.cfg
		h, w, c = img.shape

		image = transform(image=img)['image']
		image = (image / 255. - self.mean) / self.std

		image = self.img2tensor(image)

		return image		
		



	def __getitem__(self, idx):
		df_index = self.df_indexs[idx]
		info = self.df.iloc[df_index]
		
  
		image_name, label = info['file_path'], int(info['class_id'])
		observation_id = info['observation_id']
		endemic = info['endemic']
		country = info['country']
		code = info['code']
		endemic = 1 if endemic else 0


		code = int(self.code2label[code])
		country = int(self.country2label[country])
		

		image_path = os.path.join(self.image_dir, image_name)
		img = cv2.imread(image_path)
		#print(img.shape)

		image = self.transforms(image=img)['image']
		image = (image / 255. - self.mean) / self.std

		image = self.img2tensor(image)


		TTA_images = 1

		
		meta = {}
		meta['endemic'] = endemic
		meta['code'] = code
		meta['country'] = country

		item = {}

		item['image'] = image
		item['label'] = label
		item['df_index'] = df_index
		item['observation_id'] = observation_id
		item['meta'] = meta

		item['TTA_images'] = TTA_images

		return item
   

	def img2tensor(self, img):
		img = np.transpose(img, (2, 0, 1)).astype(np.float32)
		img = torch.from_numpy(img)
		return img		


	def __len__(self):
		return len(self.df_indexs)












