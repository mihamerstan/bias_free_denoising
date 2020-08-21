import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure.simple_metrics import compare_psnr
from skimage.measure import compare_ssim

import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.optim as optim
import torch

import math
import numbers
import time
import re



def batch_PSNR(img, imclean, data_range):
	Img = img.data.cpu().numpy().astype(np.float32)
	Iclean = imclean.data.cpu().numpy().astype(np.float32)
	PSNR = 0
	for i in range(Img.shape[0]):
		PSNR += compare_psnr(Iclean[i,:,:], Img[i,:,:], data_range=data_range)
	return (PSNR/Img.shape[0])

# def batch_SSIM(img, imclean, data_range):
# 	Img = img.data.cpu().numpy().astype(np.float32)
# 	Iclean = imclean.data.cpu().numpy().astype(np.float32)
# 	SSIM = 0
# 	for i in range(Img.shape[0]):
# 		SSIM += compare_ssim(Iclean[i,0,:,:], Img[i,0,:,:], data_range=data_range)
# 	return (SSIM/Img.shape[0])


def validation_error_psnr(model, dataset_val, device = torch.device('cpu')):
	model.eval()
	error_val = 0;
	psnr_val = 0;
	
	total_length = 0;

	criterion = nn.MSELoss();

	for data_x, data_y in dataset_val:
		data = data_x.to(device)
		target = data_y.to(device);
		
		out_val = model( data );

		error_val += criterion(out_val, target).item() * data.shape[0]
		psnr_val += ( batch_PSNR(out_val, target, 1.0) * data.shape[0] )
		
		total_length += data.shape[0];

	return error_val/total_length, psnr_val/total_length




class sum_squared_error(_Loss):  # PyTorch 0.4.1
	"""
	Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
	The backward is defined as: input-target
	"""
	def __init__(self, size_average=None, reduce=None, reduction='sum'):
		super(sum_squared_error, self).__init__(size_average, reduce, reduction)

	def forward(self, input, target):
		# return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
		return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)



def train_network_on_noise(model, train_loader, val_loader, 
							max_epoch = 50, lr=1e-3, plot_loss = True, verbose = False,
							verbose_freq = 1, early_stopping = True, save_plot = False, 
							filename = 'nano_denoising.pth', save_plot_address = '.', 
							saved_models_path = '.', device = torch.device('cpu') ):


	
	criterion = sum_squared_error()

	train_loss = np.zeros(max_epoch);
	val_loss = np.zeros(max_epoch);
	
	train_psnr = np.zeros(max_epoch);
	val_psnr = np.zeros(max_epoch);

	train_ssim = np.zeros(max_epoch);
	val_ssim = np.zeros(max_epoch);
	
	best_model = None;
	best_val_loss = np.inf;
	best_val_psnr = 0.0;
	
	optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) , lr=lr);
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 2, factor = 0.75, min_lr = 1e-6, verbose = True)

	
	n_batches = len(train_loader)

	initial_epoch = 0;
	for epoch in range(initial_epoch, max_epoch):
		start_time = time.time()


		epoch_train_loss = 0.0;
		epoch_train_psnr = 0.0;
		epoch_train_ssim = 0.0;
		for i, x_and_y in enumerate(train_loader):
			target = x_and_y[1].to(device);
			data = x_and_y[0].to(device);
			optimizer.zero_grad();
			model.train();

			out_train = model(data);
			
			loss = criterion(out_train, target);           
			loss.backward()
			optimizer.step()

			current_loss = loss.item();
			current_psnr = batch_PSNR(out_train, target, 1.0);
			# current_ssim = batch_SSIM(out_train, target, 1.0)
			
			if (i+1)%10 == 0:
				print('[epoch: %d][batch: %d/%d] loss: %f, psnr: %f'%(epoch, i+1, n_batches, current_loss, current_psnr))
				sys.stdout.flush()
				
			epoch_train_loss += current_loss;
			epoch_train_psnr += current_psnr;
			# epoch_train_ssim += current_ssim;

		train_loss[epoch], train_psnr[epoch] = epoch_train_loss/n_batches, epoch_train_psnr/n_batches
		val_loss[epoch], val_psnr[epoch] = validation_error_psnr(model, val_loader, 
															device = device);

		scheduler.step(val_psnr[epoch])
		
		
		elapsed_time = time.time() - start_time
		if (epoch+1) % verbose_freq == 0 and verbose:
			print(epoch+1, 'train error: ', train_loss[epoch],
							 'val error: ', val_loss[epoch], 
							  'train psnr: ', train_psnr[epoch],
							 'val psnr: ', val_psnr[epoch], 
							   'time: ', elapsed_time);
			sys.stdout.flush()

		if val_psnr[epoch] > best_val_psnr:
			best_model = model.state_dict();
			best_val_psnr = val_psnr[epoch]

			torch.save(model.state_dict(), os.path.join( saved_models_path, filename+'net.pth') )


	if early_stopping:
		model.load_state_dict(best_model)
		torch.save(model.state_dict(), os.path.join( saved_models_path, filename+'net_earlystopping.pth') )
		
	if plot_loss or save_plot:
		plt.semilogy(np.arange(1, max_epoch+1), train_loss, label = 'train loss');
		plt.semilogy(np.arange(1, max_epoch+1), val_loss, label = 'val loss')
		plt.legend()
		if plot_loss:
			plt.show()
		if save_plot:
			plt.savefig(os.path.join(save_plot_address, filename+'error.png'))
		
		plt.plot(np.arange(1, max_epoch+1), train_psnr, label = 'train psnr');
		plt.plot(np.arange(1, max_epoch+1), val_psnr, label = 'val psnr')
		plt.legend()
		if plot_loss:
			plt.show()
		if save_plot:
			plt.savefig(os.path.join(save_plot_address, filename+'psnr.png'))

	return model