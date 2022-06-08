import random
import os
import torch
import numpy as np
import argparse
import time
from tqdm import tqdm
import pandas as pd


import torchvision.transforms as transforms
import torchvision
from pytorch_toolbelt import losses as L
import torchcontrib
#import pytorch_toolbelt


from SnakeDataLayer import SnakeDataset
from net import FGVCNet
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bisect import bisect_right



class Cutmix():
    def __init__(self, alpha=1):
        self.alpha = alpha

    def __call__(self, data):
        indices = torch.randperm(data.size(0))
        #shuffled_data = data[indices]
        #shuffled_targets = targets[indices]


        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

        return data, indices, lam


    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]





def build_model(cfg, margins=None):
	model = FGVCNet(cfg, margins=margins)
	return model


def build_dataloader(cfg, is_training, rank):
	# dataset 
	dataset = SnakeDataset(cfg, is_training=is_training)

	# sampler
	if is_training:
		sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if rank != -1 else None
	else:
		sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=False) if rank != -1 else None

	if rank != -1:
			dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.samples_per_gpu, sampler=sampler, pin_memory=False, num_workers=cfg.workers_per_gpu)
	else:
		if is_training:
			dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.samples_per_gpu, shuffle=True, pin_memory=False, num_workers=cfg.workers_per_gpu)
		else:
			dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.samples_per_gpu, shuffle=False, pin_memory=False, num_workers=cfg.workers_per_gpu)

	return dataloader



def build_optimizer_diff_meta(model, cfg):
    
	optimizer = torch.optim.AdamW([
		{'params': model.backbone.parameters(), 'lr': cfg.max_lr}, 

		{'params': model.endemic_embedding.parameters(), 'lr': cfg.max_lr * 10}, 
		{'params': model.code_embedding.parameters(), 'lr': cfg.max_lr * 10},
		{'params': model.country_embedding.parameters(), 'lr': cfg.max_lr * 10},
		{'params': model.neck_meta.parameters(), 'lr': cfg.max_lr * 10},

		{'params': model.ArcHead_id_meta.parameters(), 'lr': cfg.max_lr * 10},
		{'params': model.bn.parameters(), 'lr': cfg.max_lr * 10},
		], weight_decay=5e-4)

	return optimizer	


def build_optimizer_diff(model, cfg):

	optimizer = torch.optim.AdamW([
		{'params': model.backbone.parameters(), 'lr': cfg.max_lr}, 
		{'params': model.ArcHead_id.parameters(), 'lr': cfg.max_lr * 10},
		{'params': model.bn.parameters(), 'lr': cfg.max_lr * 10},
		], weight_decay=5e-4)


	return optimizer	




def build_scheduler(optimizer, cfg, total_iters):
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters - cfg.warmup_iters, eta_min=1e-6)
	if cfg.warmup_iters > 0:
		scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=cfg.warmup_iters, after_scheduler=scheduler)	
	return scheduler




def print_train_status(epoch, iter, iters, acc, loss, cfg):
	cfg.logger.info('[Epoch:{epoch} iter:{iter}/{iters}] acc: {acc} loss: {loss}'.format(epoch=epoch, iter=iter, iters=iters, acc=acc, loss=loss))



def calc_acc(outputs, labels):
	_, predicted = torch.max(outputs.data, 1)
	true_num = float(predicted.eq(labels.data).cpu().sum())
	all_num = float(labels.size(0))
	acc = true_num / all_num
	return acc





def save_checkpoints(model, epoch, cfg, epoch_begin=0):
	save_checkpoint = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
	checkpoint_name = '{model_name}-fold{fold}_epoch_{epoch}'.format(model_name=cfg.backbone, fold=cfg.fold, epoch=epoch+epoch_begin)
	torch.save(save_checkpoint, os.path.join(cfg.work_dir, checkpoint_name))



def eval_acc(model, data_loader, cfg):
	true_num, all_num = 0, 0
	time_start = time.time()

	model.eval()
	with torch.no_grad():
		for data in tqdm(data_loader):
			images, labels, chains = data
			images, labels, chains = images.cuda(), labels.long().cuda(), chains.long().cuda()
			outputs = model(images.float())

			#_, predicted = torch.max(outputs[:, :7770].data, 1)
			_, predicted = torch.max(outputs.data, 1)
			all_num += labels.size(0)
			true_num += float(predicted.eq(labels.data).cpu().sum())
	time_end = time.time()
	eval_time = time_end - time_start
	acc = true_num / all_num
	cfg.logger.info('eval acc: {acc} time: {time}s'.format(acc=acc, time=eval_time))



def eval_topk(model, data_loader, cfg, max_k=5):
	true_num, all_num = 0, 0
	time_start = time.time()

	corrects = [0 for k in range(max_k)]

	model.eval()
	with torch.no_grad():
		for data in tqdm(data_loader):
			images, labels, chains = data
			images, labels, chains = images.cuda(), labels.long().cuda(), chains.long().cuda()
			if cfg.use_chain:
				outputs, _, _, _= model(images.float())
			else:
				outputs = model(images.float())

			#_, predicted = torch.max(outputs[:, :7770].data, 1)
			#_, predicted = torch.max(outputs.data, 1)=
			_, pred = outputs.topk(max_k, 1, True, True)
			pred = pred.t()
			correct = pred.eq(labels.view(1, -1).expand_as(pred))

			

			for k in range(1, max_k+1):
				correct_k = correct[:k, :].contiguous().view(-1).float().sum(0, keepdim=True)
				corrects[k-1] += correct_k.item()

			all_num += labels.size(0)
			#true_num += float(predicted.eq(labels.data).cpu().sum())
	time_end = time.time()
	eval_time = time_end - time_start

	rank_k_acc = [correct / all_num for correct in corrects]


	#acc = true_num / all_num
	cfg.logger.info(f'eval rank-1: {rank_k_acc[0]:.3f}  rank-2: {rank_k_acc[1]:.3f} rank-3: {rank_k_acc[2]:.3f} rank-4: {rank_k_acc[3]:.3f} rank-5: {rank_k_acc[4]:.3f}  time: {eval_time:.0f}s')













