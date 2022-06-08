import random
import os
from cv2 import KeyPoint, sortIdx
import torch
import numpy as np
import argparse
import time
from tqdm import tqdm
import pandas as pd
import torch.distributed as dist

from sklearn.metrics import f1_score 
from terminaltables import AsciiTable
from torch_scatter import scatter     

def eval_dist(model, data_loader, cfg, max_k=5):
	time_start = time.time()
	world_size = dist.get_world_size()
	all_labels_, all_pred_labels_, all_pred_labels_topk_, all_image_names_, all_observation_ids_, all_outputs_ = eval_single(model, data_loader, cfg)

	all_labels, all_pred_labels, all_pred_labels_topk, all_image_names, all_observation_ids, all_outputs = [[None] * world_size for i in range(6)]

	dist.all_gather_object(all_labels, all_labels_)
	dist.all_gather_object(all_pred_labels, all_pred_labels_)
	dist.all_gather_object(all_pred_labels_topk, all_pred_labels_topk_)
	dist.all_gather_object(all_image_names, all_image_names_)
	dist.all_gather_object(all_observation_ids, all_observation_ids_)
	dist.all_gather_object(all_outputs, all_outputs_)

	all_labels = torch.cat(all_labels, dim=0)
	all_pred_labels = torch.cat(all_pred_labels, dim=0)
	all_pred_labels_topk = torch.cat(all_pred_labels_topk, dim=1)
	all_image_names = torch.cat(all_image_names, dim=0)
	all_observation_ids = torch.cat(all_observation_ids, dim=0)
	all_outputs = torch.cat(all_outputs, dim=0)

	eval_results(cfg, all_pred_labels, all_labels, all_image_names, all_pred_labels_topk, all_observation_ids, all_outputs)

	time_end = time.time()
	eval_time = time_end - time_start
	cfg.logger.info(f'eval time: {eval_time:.0f}s')


def eval_single(model, data_loader, cfg, max_k=5):
	all_labels = []
	all_pred_labels = []
	all_image_names = []
	all_pred_labels_topk = []
	all_observation_ids = []
	all_outputs = []

	model.eval()
	with torch.no_grad():
		for batch, data in enumerate(data_loader):
			images, labels, meta, image_names = data['image'], data['label'], data['meta'], data['df_index']
			observation_ids = data['observation_id'].numpy().tolist()
			images, labels = images.cuda(), labels.long().cuda()
			
			for key in meta.keys():
				meta[key] = meta[key].cuda()
				
			outputs = model(images.float(), meta=meta)[0]

			all_outputs.append(outputs.cpu().detach())

			# top k
			_, pred = outputs.topk(max_k, 1, True, True)
			pred = pred.t()
   
			_, pred_labels = torch.max(outputs, 1)
			all_labels.extend(labels.cpu().numpy().tolist())
			all_pred_labels.extend(pred_labels.cpu().numpy().tolist())
			all_image_names.extend(image_names)
			all_pred_labels_topk.append(pred)
			all_observation_ids.extend(observation_ids)

		all_pred_labels_topk = torch.cat(all_pred_labels_topk, dim=1).cpu().detach()
		all_outputs = torch.cat(all_outputs, dim=0)

		all_labels = torch.tensor(all_labels)
		all_pred_labels = torch.tensor(all_pred_labels)
		all_image_names = torch.tensor(all_image_names)
		all_observation_ids = torch.tensor(all_observation_ids)



	return all_labels, all_pred_labels, all_pred_labels_topk, all_image_names, all_observation_ids, all_outputs



def scatter_all_outputs(all_labels, all_outputs, all_observation_ids, max_k=5):
	#print('scatter_outputs')
	N = len(all_observation_ids)

	# sort
	sort_index = np.argsort(all_observation_ids)
	all_outputs = all_outputs[sort_index, :]
	all_observation_ids = all_observation_ids[sort_index]
	all_labels = all_labels[sort_index]

	unique_index = np.unique(all_observation_ids, return_index=True)[1]

	

	inds = np.zeros(shape=N, dtype=np.int64)	

	ind = 0
	for i in range(len(unique_index)-1):
		start = unique_index[i]
		num = unique_index[i+1] - start
		inds[start:start+num] = ind
		ind += 1
	inds[unique_index[-1]:] = ind	

	scatter_outputs = scatter(torch.tensor(all_outputs), torch.tensor(inds), dim=0, reduce='mean')
	all_labels = all_labels[unique_index]


	_, all_pred_labels_topk = scatter_outputs.topk(max_k, 1, True, True)
	all_pred_labels_topk = all_pred_labels_topk.t()

	all_pred_probs, all_pred_labels = torch.max(scatter_outputs, 1)

	#all_pred_labels[all_pred_probs < 0.1] = -1

	return all_labels, all_pred_labels, all_pred_labels_topk



def eval_results(cfg, all_pred_labels, all_labels, all_image_names, all_pred_labels_topk, all_observation_ids, all_outputs, max_k=5):
	N = all_pred_labels.shape[0]

	keep = np.array([False] * N)
	all_image_names = all_image_names.numpy()

	keep[np.unique(all_image_names, return_index=True)[1]] = True
	

	all_labels = all_labels[keep]
	all_pred_labels = all_pred_labels[keep]
	all_pred_labels_topk = all_pred_labels_topk[:, keep]
	all_outputs = all_outputs[keep, :]
	all_observation_ids = all_observation_ids[keep]


	

	N = all_labels.shape[0]
	
	corrects = [0 for k in range(max_k)]
	correct = all_pred_labels_topk.eq(all_labels.view(1, -1).expand_as(all_pred_labels_topk))

	for k in range(1, max_k+1):
		correct_k = correct[:k, :].contiguous().view(-1).float().sum(0, keepdim=True)
		corrects[k-1] += correct_k.item()

	


	mean_f1_score = f1_score(all_labels, all_pred_labels, average='macro') * 100
   
	


	rank_k_acc = [correct / N for correct in corrects]
	rank_k_acc = [val * 100 for val in rank_k_acc]

	#acc = true_num / all_num
	#cfg.logger.info(f'eval rank-1: {rank_k_acc[0]:.3f}  rank-3: {rank_k_acc[2]:.3f}   time: {eval_time:.0f}s')

	# 表格可视化
	class_table_data = [['f1_score', 'rank-1', 'rank-3']]

	class_table_data.append([f'{mean_f1_score:.2f}'] + [f'{rank_k_acc[0]:.2f}'] + [f'{rank_k_acc[2]:.2f}'])
	

	all_labels, all_pred_labels, all_pred_labels_topk = scatter_all_outputs(all_labels, all_outputs, all_observation_ids)


	N = all_labels.shape[0]
	
	corrects = [0 for k in range(max_k)]
	correct = all_pred_labels_topk.eq(all_labels.view(1, -1).expand_as(all_pred_labels_topk))

	for k in range(1, max_k+1):
		correct_k = correct[:k, :].contiguous().view(-1).float().sum(0, keepdim=True)
		corrects[k-1] += correct_k.item()

	


	mean_f1_score = f1_score(all_labels, all_pred_labels, average='macro') * 100
   
	


	rank_k_acc = [correct / N for correct in corrects]
	rank_k_acc = [val * 100 for val in rank_k_acc]
	class_table_data.append([f''] + [f''] + [f''])
	class_table_data.append([f'{mean_f1_score:.2f}'] + [f'{rank_k_acc[0]:.2f}'] + [f'{rank_k_acc[2]:.2f}'])


	cfg.logger.info('')
	table = AsciiTable(class_table_data)
	cfg.logger.info(table.table)
	cfg.logger.info('')	



	





def eval_fung(model, data_loader, cfg, max_k=5):
	true_num, all_num = 0, 0
	time_start = time.time()

	corrects = [0 for k in range(max_k)]
 
	all_labels = []
	all_pred_labbels = []

	model.eval()
	with torch.no_grad():
		for batch, data in enumerate(tqdm(data_loader)):
			images, labels, meta = data
			images, labels = images.cuda(), labels.long().cuda()
			
			#for key in meta.keys():
			#	meta[key] = meta[key].cuda()


			outputs = model(images.float(), meta)[0]

			#_, predicted = torch.max(outputs[:, :7770].data, 1)
			#_, predicted = torch.max(outputs.data, 1)=
   
			_, pred_labels = torch.max(outputs, 1)
			all_labels.extend(labels.cpu().numpy().tolist())
			all_pred_labbels.extend(pred_labels.cpu().numpy().tolist())
   
			_, pred = outputs.topk(max_k, 1, True, True)
			pred = pred.t()
			
			correct = pred.eq(labels.view(1, -1).expand_as(pred))
			
			

			for k in range(1, max_k+1):
				correct_k = correct[:k, :].contiguous().view(-1).float().sum(0, keepdim=True)
				corrects[k-1] += correct_k.item()

			all_num += labels.size(0)
			#true_num += float(predicted.eq(labels.data).cpu().sum())

			#if batch > 5:
			#	break
   
	all_labels = np.array(all_labels)
	all_pred_labbels = np.array(all_pred_labbels)
	
	mean_f1_score = f1_score(all_labels, all_pred_labbels, average='macro') * 100
   
   
	time_end = time.time()
	eval_time = time_end - time_start

	rank_k_acc = [correct / all_num for correct in corrects]
	rank_k_acc = [val * 100 for val in rank_k_acc]

	#acc = true_num / all_num
	#cfg.logger.info(f'eval rank-1: {rank_k_acc[0]:.3f}  rank-3: {rank_k_acc[2]:.3f}   time: {eval_time:.0f}s')

	# 表格可视化
	class_table_data = [['f1_score', 'rank-1', 'rank-3']]

	class_table_data.append([f'{mean_f1_score:.2f}'] + [f'{rank_k_acc[0]:.2f}'] + [f'{rank_k_acc[2]:.2f}'])
	
	cfg.logger.info('')
	table = AsciiTable(class_table_data)
	cfg.logger.info(table.table)
	cfg.logger.info('')	

	cfg.logger.info(f'eval time: {eval_time:.0f}s')











