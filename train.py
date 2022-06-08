import warnings
warnings.filterwarnings("ignore")

from config import *

# mmcls
from utils.util import *
from util import Cutmix


from torch.cuda import amp


import torch 

from util import *
from eval_util import *




def train(args, cfg):

	if args.local_rank in [-1, 0]:
		cfg.logger.info(f'local_rank: {args.local_rank}')

	#### build data loader
	train_loader = build_dataloader(cfg, is_training=True, rank=args.local_rank)
	val_loader = build_dataloader(cfg, is_training=False, rank=args.local_rank)

	#### build model
	model = build_model(cfg, train_loader.dataset.margins)


	model_path = args.checkpoint_path
	if model_path != '':
		model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
	model = model.cuda()


	#### build optimizer
	if cfg.use_meta:
		optimizer = build_optimizer_diff_meta(model, cfg)
	else:
		optimizer = build_optimizer_diff(model, cfg)

	#### build scheduler
	total_iters = cfg.max_epoch * len(train_loader)
	scheduler = build_scheduler(optimizer, cfg, total_iters+1)
	scheduler.step()

	# sync
	if args.local_rank != -1:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

	# DistributedDataParallel
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
		model._set_static_graph()


	iters = int(len(train_loader))

	if cfg.use_amp:
		scaler = amp.GradScaler()


	if cfg.use_cutmix:
		cutmix = Cutmix(alpha=1)



	for epoch in range(1, cfg.max_epoch + 1):
		model.train()

		if args.local_rank != -1:
			train_loader.sampler.set_epoch(epoch)


		cfg.logger.info('')
		cfg.logger.info('epoch: {epoch} lr: {lr}'.format(epoch=epoch, lr=optimizer.param_groups[0]['lr']))

		time_start = time.time()

		# train
		for i, data in enumerate(train_loader):
			scheduler.step()
			use_cutmix = False

			# data
			images, targets, meta = data['image'], data['label'], data['meta']
			
			images, targets = images.cuda().float(), targets.long().cuda()

			for key in meta.keys():
				meta[key] = meta[key].cuda()

			if cfg.use_cutmix and np.random.random() < cfg.cutmix_prob and epoch < int(cfg.mix_ratio * cfg.max_epoch):
				use_cutmix = True
				mix_images, mix_indices, lam = cutmix(images)

			## forward loss
			if cfg.use_amp:
				with amp.autocast():
					if use_cutmix:
						outputs_id, loss = model(mix_images, targets, meta, lam, mix_indices)
					else:
						outputs_id, loss = model(images, targets, meta)
      
			# visualize
			if i % 200 == 0:
				acc = calc_acc(outputs_id, targets)
				print_train_status(epoch, i, iters, acc, loss.item(), cfg)

			ni = i + iters * epoch

			if cfg.use_amp:
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()
			else:
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

		time_end = time.time()
		cfg.logger.info('train time: {time}s'.format(time=time_end - time_start))


		if epoch % 5 == 0 or epoch == 1 or epoch == cfg.max_epoch:
			if epoch % 5 == 0 or epoch == cfg.max_epoch:
				save_checkpoints(model, epoch, cfg)

			eval_dist(model, val_loader, cfg)





if __name__ == '__main__':

	seed_everything(42)
	cv2.setNumThreads(0)

	parser = argparse.ArgumentParser()	
	parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
	parser.add_argument('--name', default='mmcls123', type=str)

	parser.add_argument('--checkpoint_path', default='', type=str)
	parser.add_argument('--backbone', default='', type=str)
	parser.add_argument('--use_meta', action='store_true')
	parser.add_argument('--fix_backbone', action='store_true')
	parser.add_argument('--max_epoch', default=20, type=int)
	parser.add_argument('--max_lr', default=1e-4, type=float)
	parser.add_argument('--samples_per_gpu', default=32, type=int)
	parser.add_argument('--image_size', default=512, type=int)

	
	
	
	args = parser.parse_args()	


	if args.local_rank != -1:
		dist.init_process_group(backend='nccl')
		torch.cuda.set_device(args.local_rank)
	
	if args.local_rank == -1:
		cfg = Cfg()

		cfg.backbone = args.backbone
		cfg.use_meta = args.use_meta
		cfg.fix_backbone = args.fix_backbone
		cfg.max_epoch = args.max_epoch
		cfg.max_lr = args.max_lr
		cfg.samples_per_gpu = args.samples_per_gpu
		cfg.image_size = args.image_size

		cfg.work_dir, cfg.logger = init_logger(cfg)
		print_cfg(cfg.logger, cfg)
	else:
		with torch_distributed_zero_first(args.local_rank):
			cfg = Cfg()

			cfg.backbone = args.backbone
			cfg.use_meta = args.use_meta
			cfg.fix_backbone = args.fix_backbone
			cfg.max_epoch = args.max_epoch
			cfg.max_lr = args.max_lr
			cfg.samples_per_gpu = args.samples_per_gpu
			cfg.image_size = args.image_size

			cfg.work_dir, cfg.logger = init_logger(cfg, args.local_rank)
			if args.local_rank == 0:
				print_cfg(cfg.logger, cfg)
	train(args, cfg)














