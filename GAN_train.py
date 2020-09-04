import argparse
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from utils import data
import models, utils


def main(args):
	# gpu or cpu
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	utils.setup_experiment(args)
	utils.init_logging(args)

	### Define models and optimizers
	# Build data loaders, a model and an optimizer
	G,D = models.build_model_gan(args)
	netG = G.to(device)
	netD = D.to(device)


	# custom weights initialization called on netG and netD
	def weights_init(m):
	    classname = m.__class__.__name__
	    if classname.find('Conv') != -1:
	        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	    elif classname.find('BatchNorm') != -1:
	        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
	        torch.nn.init.constant_(m.bias.data, 0)

	netG.apply(weights_init)
	netD.apply(weights_init)

	# 2 optimizers
	optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lrG,betas=(0.5, 0.999))
	optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lrD,betas=(0.5, 0.999))

	# Schedulers to reduce the learning rate
	schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, 10, gamma=0.5)
	schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, 10, gamma=0.5)

	logging.info(f"Built a generator model consisting of {sum(p.numel() for p in netG.parameters()):,} parameters")
	logging.info(f"Built a discriminator model consisting of {sum(p.numel() for p in netD.parameters()):,} parameters")

	if args.resume_training and args.restore_file is not None:
	    state_dict = utils.load_checkpoint_GAN(args, netG, netD, optimizerG, optimizerD)
	    global_step = state_dict['last_step']
	    start_epoch = int(state_dict['last_step']/(403200/state_dict['args'].batch_size))+1
	else:
	    global_step = -1
	    start_epoch = 0

	# build_dataset is a function in utils/data/__init__.py
	train_loaderG, valid_loaderG, _ = data.build_dataset(args.datasetG,
	                                                   args.n_data, 
	                                                   batch_size=args.batch_size,
	                                                   min_sep = args.min_sep)
	train_loaderD, _, _ = data.build_dataset(args.datasetD,
	                                                   args.n_data, 
	                                                   batch_size=args.batch_size,
	                                                   min_sep = args.min_sep)

	# Initialize BCELoss function
	criterion = torch.nn.BCELoss()

	# Track moving average of loss values
	train_meters = {name: utils.RunningAverageMeter(0.98) for name in ([\
	                                    "G_loss","G_loss_L2","G_loss_adv","accD","accD_fake","accD_real"])}
	# valid_meters = {name: utils.AverageMeter() for name in (["Loss"])}
	writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None

	# TRAINING
	fake_label = 0.
	real_label = 1.

	G_losses = []
	D_losses = []

	# Determine whether or not MSE is being used, which alters the loss reporting
	if args.wtl2 > 0:
	    use_mse = True
	else:
	    use_mse = False

	for epoch in range(start_epoch, args.num_epochs):
	    if args.resume_training:
	        if epoch %10 == 0:
	            optimizerG.param_groups[0]["lr"] /= 2
	            print('learning rate reduced by factor of 2')

	    train_bar = utils.ProgressBar(train_loaderG, epoch)
	    for meter in train_meters.values():
	        meter.reset()

	    for batch_id, ((clean, mask),real) in enumerate(zip(train_bar,train_loaderD)):
	        
	        ###############################
	        # First train the discriminator
	        ###############################
	        # Only update discriminator based on update ratio
	        real_cpu = real.to(device)
	        b_size = real_cpu.size(0)
	        label = torch.full((b_size,),real_label,device=device)

	        if batch_id % args.g_d_update_ratio == 0:
	            netD.zero_grad()

	            # Forward pass real batch through D
	            output = netD(real_cpu).view(-1)
	            # Calculate loss on all-real batch
	            errD_real = criterion(output, label)
	            # Calculate gradients for D in backward pass
	            errD_real.backward()
	            D_x = output.mean().item()
	            # Accuracy is  the number of 1's when all samples are real
	            preds = [1 if o > 0.5 else 0 for o in output]
	            accD_real = np.sum(preds)/b_size
	        
	        ## Train with all-fake batch
	        # Generate fake signal batch with G
	        inputs = clean.to(device)
	        mask_inputs = mask.to(device)
	        # only use the mask part of the outputs
	        raw_outputs = netG(inputs,mask_inputs)
	        fake = (1-mask_inputs)*raw_outputs + mask_inputs*inputs
	        
	        label.fill_(fake_label)
	        if batch_id % args.g_d_update_ratio == 0:
	            # Classify all fake batch with D
	            output = netD(fake.detach()).view(-1)
	            # Calculate D's loss on the all-fake batch
	            errD_fake = criterion(output, label)

	            # Accuracy is  the number of 1's when all samples are fake
	            # Note I flipped the pred logic for fakes
	            preds = [1 if o < 0.5 else 0 for o in output]
	            accD_fake = np.sum(preds)/b_size
	            accD = (accD_fake+accD_real)/2
	            # Calculate the gradients for this batch
	            errD_fake.backward()
	            D_G_z1 = output.mean().item()
	            # Add the gradients from the all-real and all-fake batches
	            errD = errD_real + errD_fake

	            # Update D
	            optimizerD.step()
	        
	        ###############################
	        # Next, train the generator
	        ###############################
	        netG.zero_grad() # train() or zero_grad()?
	        label.fill_(real_label)  # fake labels are real for generator cost
	        # Since we just updated D, perform another forward pass of all-fake batch through D
	        output = netD(fake).view(-1)
	        
	        if use_mse:
	            # Calculate G's loss based on this output
	            errG_D = criterion(output, label)
	            # MSE Loss
	            errG_l2 = F.mse_loss(fake, inputs, reduction="sum") / (inputs.size(0) * 2)
	            errG =  (1-args.wtl2) * errG_D + args.wtl2 * errG_l2
	        else:
	            errG_l2 = torch.zeros(1)
	            errG = criterion(output, label)
	            errG_D = errG
	        # Calculate gradients for G
	        errG.backward()
	        D_G_z2 = output.mean().item()
	        # Update G
	        optimizerG.step()
	        

	        
	        # Output training stats
	        if batch_id % 50 == 0:
	            if use_mse:
	                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f + %.4f = %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
	                  % (epoch, args.num_epochs, batch_id, len(train_loaderG), \
	                     errD.item(), errG_D.item(), errG_l2, errG.item(), D_x, D_G_z1, D_G_z2))
	            else: 
	                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
	                  % (epoch, args.num_epochs, batch_id, len(train_loaderG),
	                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

	        # Save Losses for plotting later
	        G_losses.append(errG.item())
	        D_losses.append(errD.item())

	        global_step += 1
	        
	        # TO DO, only run loss on masked part of output
	        # TO DO, incorporate MSE loss into GAN?
	        # loss = F.mse_loss(outputs, inputs, reduction="sum") / (inputs.size(0) * 2)
	        
	        # New train_meters
	        train_meters["G_loss"].update(errG.item())
	        if use_mse:
	            train_meters["G_loss_L2"].update(errG_l2.item())
	            train_meters["G_loss_adv"].update(errG_D.item())
	        else:
	            train_meters["G_loss_L2"].update(0)
	            train_meters["G_loss_adv"].update(0)
	        train_meters["accD"].update(accD.item())
	        train_meters["accD_fake"].update(accD_fake.item())
	        train_meters["accD_real"].update(accD_real.item())

	        train_bar.log(dict(**train_meters, lr=optimizerG.param_groups[0]["lr"]), verbose=True)

	        if writer is not None and global_step % args.log_interval == 0:
	            print("Got into the writer!: {} {}".format(errG.item(),accD_real.item()))
	            writer.add_scalar("lr", optimizerG.param_groups[0]["lr"], global_step)
	            writer.add_scalar("G_loss/train", errG.item() , global_step)
	            writer.add_scalar("G_loss_L2/train", errG_l2.item() , global_step)
	            writer.add_scalar("G_loss_adv/train", errG_D.item() , global_step)
	            writer.add_scalar("accD/train", accD.item() , global_step)
	            writer.add_scalar("accD_fake/train", accD_fake.item() , global_step)
	            writer.add_scalar("accD_real/train", accD_real.item() , global_step)
	            gradients = torch.cat([p.grad.view(-1) for p in netG.parameters() if p.grad is not None], dim=0)
	            writer.add_histogram("gradients", gradients, global_step)
	            sys.stdout.flush()

	# Rewrite validation loop for the right metrics
	    # if epoch % args.valid_interval == 0:
	#         netG.eval()
	#         for meter in valid_meters.values():
	#             meter.reset()

	#         valid_bar = utils.ProgressBar(valid_loaderG)
	        
	#         for sample_id, (clean, mask) in enumerate(valid_bar):
	#             with torch.no_grad():
	#                 inputs = clean.to(device)
	#                 mask_inputs = mask.to(device)
	#                 # only use the mask part of the outputs
	#                 raw_output = netG(inputs,mask_inputs)
	#                 output = (1-mask_inputs)*raw_output + mask_inputs*inputs
	#                 valid_psnr = utils.psnr(inputs, output)
	#                 valid_meters["valid_psnr"].update(valid_psnr.item())
	#                 valid_ssim = utils.ssim(inputs, output)
	#                 valid_meters["valid_ssim"].update(valid_ssim.item())

	#         if writer is not None:
	#             writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, global_step)
	#             writer.add_scalar("ssim/valid", valid_meters['valid_ssim'].avg, global_step)
	#             sys.stdout.flush()
	#             utils.save_checkpoint_GAN(args, global_step, netG, netD, optimizerG, optimizerD, score=valid_meters["valid_psnr"].avg, mode="max")

	    if epoch % args.log_interval == 0:
	        logging.info(train_bar.print(dict(**train_meters,lr=optimizerG.param_groups[0]["lr"])))

	    schedulerG.step()
	    schedulerD.step()
	# Save the loss curve plot
	utils.save_losses_curve(G_losses,D_losses,args)
	# logging.info(f"Done training! Best PSNR {utils.save_checkpoint_GAN.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")
	logging.info(f"Done training!")


def get_args():
	parser = argparse.ArgumentParser(allow_abbrev=False)

	# Add data arguments
	parser.add_argument("--data-path", default="data", help="path to data directory")
	parser.add_argument("--datasetG", default="masked_pwc", help="masked training data for generator")
	parser.add_argument("--datasetD", default="pwc", help="unmasked training data for generator")
	parser.add_argument("--batch-size", default=256, type=int, help="train batch size")
	parser.add_argument("--n-data", default=100000,type=int, help="number of samples")
	parser.add_argument("--min_sep", default=5,type=int, help="minimum constant sample count for piecwewise function")


	# Add model arguments
	parser.add_argument("--model", default="unet1d", help="MSE model architecture (should remove this)")
	parser.add_argument("--modelG", default="unet1d", help="Generator model architecture")
	parser.add_argument("--modelD", default="gan_discriminator", help="Discriminator model architecture")
	parser.add_argument("--wtl2", default="0.", type=float, help="weighting to L2 loss, remainder is adversarial loss")
	parser.add_argument("--g_d_update_ratio", default = 2, type=int, help="How many times to update G for each update of D")

	# Add optimization arguments
	parser.add_argument("--lrG", default=.0004, type=float, help="learning rate for generator")
	parser.add_argument("--lrD", default=.001, type=float, help="learning rate for discriminator")
	parser.add_argument("--num-epochs", default=1000, type=int, help="force stop training at specified epoch")
	parser.add_argument("--valid-interval", default=25, type=int, help="evaluate every N epochs")
	parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")

	# Parse twice as model arguments are not known the first time
	parser = utils.add_logging_arguments(parser)
	args, _ = parser.parse_known_args()
	models.MODEL_REGISTRY[args.modelG].add_args(parser)
	models.MODEL_REGISTRY[args.modelD].add_args(parser)
	args = parser.parse_args()
	print("vars(args)",vars())
	return args


if __name__ == "__main__":
	args = get_args()
	main(args)

