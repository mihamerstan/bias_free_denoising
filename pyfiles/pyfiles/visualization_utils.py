import numpy as np
import matplotlib.pyplot as plt
import training_utils
import jacobian_utils
# import nnet_models
import torch
from functools import partial
# import affine_approx_around_source

def get_error(array1, array2):
	assert( len(array1) == len(array2));
	return np.mean( (array1 - array2)**2 )


def visualize_denoising(source, net, add_noise = False, noise_std = 0.1, add_blur = False, blur_sigma = 1, blur_kernel_size = 9,  
							figsize=(10,10), x_in = None, second_net = None):
	
	if x_in is None:
		x_in = source.clone();

		if add_blur:
			x_in = training_utils.blur_with_gaussian(x_in, kernel_size = blur_kernel_size, sigma = blur_sigma);

		if add_noise:
			noise_sample = training_utils.get_noise(x_in, noise_std)
			x_in = x_in + noise_sample;

	reconstructed = net(x_in);
	if second_net is not None:
		reconstructed_second_net = second_net(x_in);
		reconstructed_second_net = reconstructed_second_net.data.cpu().numpy().reshape(-1);
	
	
	corrupted_psnr = round( training_utils.batch_PSNR(x_in, source, 1.0), 2);
	reconstructed_psnr = round( training_utils.batch_PSNR(reconstructed, source, 1.0), 2)
	source = source.cpu().data.numpy().reshape(-1);
	# noise_sample = noise_sample.data.cpu().numpy().reshape(-1);
	x_in = x_in.data.cpu().numpy().reshape(-1);
	reconstructed = reconstructed.data.cpu().numpy().reshape(-1);
	
	fig, axes = plt.subplots(1, 3 + int(second_net is not None), sharex=True, sharey=True, figsize=figsize);
	axes[0].plot(source)
	axes[0].set_title('Signal')
	
#     axes[1].plot(noise_sample)
#     axes[1].set_title('Noise Sample')
	
	axes[1].plot(x_in, 'b', label='corrupted')
	axes[1].plot(source, 'r--', label='source')
	axes[1].legend()
	axes[1].set_title('corrupted. PSNR: '+str( corrupted_psnr  ) )
	
#     axes[3].plot(reconstructed)
#     axes[3].set_title('Reconstructed. Error: '+str( get_error(reconstructed, source)  ) )
	
	axes[2].plot(reconstructed,'b', label='recon')
	axes[2].plot(source, 'r--', label='source')
	axes[2].legend()
	axes[2].set_title('Reconstructed. PSNR: '+str( reconstructed_psnr  ) )

	if second_net is not None:
		axes[3].plot(reconstructed_second_net,'b', label='recon')
		axes[3].plot(source, 'r--', label='source')
		axes[3].legend()
		axes[3].set_title('Reconstructed 2nd Net. Error: '+str( get_error(reconstructed_second_net, source)  ) )


def plot_singular_values(s, figsize=(20, 5) ):
	fig, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=figsize);
	axes[0].plot(s)
	axes[0].set_title('All Singular Values')
	
	axes[1].plot(s[:10])
	axes[1].set_title('Top 10 Singular Values')
	
	axes[2].semilogy(s)
	axes[2].set_title('Singular Values in log scale')


def plot_singular_vectors(u, k = 5, figsize=(20, 5), top = True ):
	## Assume vectors are along column
	
	fig, axes = plt.subplots(1, k, sharex=True, sharey=True, figsize=figsize);
	
	n = u.shape[1]
	
	for i in range(k):
		if top:
			axes[i].plot(u[:, i]);
			axes[i].set_title('Top : ' +str(i+1))
		else:
			axes[i].plot(u[:, n-i-1]);
			axes[i].set_title('Bottom : ' + str(i+1))

def project_to_u_s_vt(u, s, vt, x):
	out = np.dot(vt, x);
	out = np.dot(np.diag(s), out);
	out = np.dot(u, out);
	return out


def all_projection_plots(u, s, vt, x, bias, one_sing_idx, figsize=(20, 5)):

	fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=figsize);
	
	k = 5
	proj_to_k = project_to_u_s_vt(u[:, :k], s[:k], vt[:k, :], x);
	axes[0][0].plot(proj_to_k)
	axes[0][0].set_title('projection to top \n' + str(k) +' components')
	
	
	axes[0][1].plot(proj_to_k + bias)
	axes[0][1].set_title('projection to top \n' + str(k) +' components + bias')
	
	
	
	proj_to_1_to_k = project_to_u_s_vt(u[:, 1:k], s[1:k], vt[1:k, :], x);
	axes[0][2].plot(proj_to_1_to_k)
	axes[0][2].set_title('proj to top \n' + str(1) + ':' + str(k) +' components')
	

	proj_to_k = project_to_u_s_vt(u[:, one_sing_idx], s[one_sing_idx],
								  vt[one_sing_idx, :], x);
	axes[1][0].plot(proj_to_k, label='wo bias')
	axes[1][0].plot(proj_to_k + bias, label='w bias')
	axes[1][0].legend()
	axes[1][0].set_title('proj to sing val \n 1 directions + bias')
	
	
	proj_to_k = project_to_u_s_vt(u[:, :1], s[:1],
								  vt[:1, :], x);
	axes[1][1].plot(proj_to_k)
	axes[1][1].set_title('proj to first dir')
	
	axes[1][2].plot(proj_to_k + bias)
	axes[1][2].set_title('proj to first dir \n + bias')
	

def plot_corr_with_u_and_v(source, noise_sample, u, vt, figsize = (20, 5)):
	source_numpy = source.clone().cpu().data.numpy().reshape(-1, 1);
	source_numpy /= np.linalg.norm(source_numpy)

	noise_sample_numpy = noise_sample.clone().cpu().data.numpy().reshape(-1, 1);
	noise_sample_numpy /= np.linalg.norm(noise_sample_numpy)

	source_with_u = np.dot(u.T, source_numpy);
	source_with_v = np.dot(vt, source_numpy);
	noise_with_u = np.dot(u.T, noise_sample_numpy);
	noise_with_v = np.dot(vt, noise_sample_numpy);

	fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize);
	axes[0].plot(source_with_u, 'o', label='source')
	axes[0].plot(noise_with_u, '+', label = 'noise')
	axes[0].legend()
	axes[0].set_title('with u')

	axes[1].plot(source_with_v, 'o', label='source')
	axes[1].plot(noise_with_v, '+', label = 'noise')
	axes[1].legend()
	axes[1].set_title('with v')


def get_jump_points(signal):
    jump_points = [];
    prev = signal[0];
    for i in range(1, len(signal)):
        if signal[i] != prev:
            jump_points.append(i)
            prev = signal[i]
    return jump_points

def axvlines(xs, ax=None, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param ax: The axis (or none to use gca)
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scaley = False, **plot_kwargs)


def plot_rows_of_jac(jac, jac_norm = None, jump_points = None, n_cols = 5):

	if (np.sum( np.square( jac)) == 0):
		print('All Zeros in this jacobian')
		return

	n_rows = jac.shape[0] // n_cols;
	if jac.shape[0]%n_cols != 0:
		n_rows += 1;

	for i in range(n_rows):
		fig, axes = plt.subplots(1 + int(jac_norm is not None), n_cols, sharex=False, sharey=False, figsize=(25, 5 + 5*int(jac_norm is not None)));
		for j in range(n_cols):
			if (i*n_cols + j) >= jac.shape[0]:
				if jac_norm is None:
					axes[j].axis('off')
				else:
					axes[0, j].axis('off')
					axes[1, j].axis('off')
			else:
				if jac_norm is None:
					axes[j].stem(jac[i*5 + j, :]);
					axes[j].set_title('gradf '+str(i*5 + j + 1) + ' sum: ' + str( np.sum(jac[i*5 + j, :]) ) )
					axvlines([i*n_cols + j], axes[j],  linestyle = 'dashed', color = 'k')

					if jump_points is not None:
						axvlines(jump_points, axes[j],  linestyle = 'dashed', color = 'r')
				else:
					axes[0, j].stem(jac[i*5 + j, :]);
					axes[0, j].set_title('gradf '+str(i*5 + j + 1) + ' sum: ' + str( np.sum(jac[i*5 + j, :]) ) )
					axvlines([i*n_cols + j], axes[0, j],  linestyle = 'dashed', color = 'k')

					axes[1, j].stem(jac_norm[i*5 + j, :]);
					axes[1, j].set_title('grad^2f '+str(i*5 + j + 1) + ' sum: ' + str( np.sum(jac_norm[i*5 + j, :]) ) )
					axvlines([i*n_cols + j], axes[1, j],  linestyle = 'dashed', color = 'k')

					if jump_points is not None:
						axvlines(jump_points, axes[0, j],  linestyle = 'dashed', color = 'r')
						axvlines(jump_points, axes[1, j],  linestyle = 'dashed', color = 'r')
		plt.show()
		print('\n'+'='*100+'\n')
	

def contribution_from_jac_and_bias(jac, bias, x_in, figsize = (20, 5), m_inv_b = False):
	x_in = x_in.cpu().data[0,0].numpy().reshape(-1, 1);
	from_jac = np.dot(jac, x_in);
	fig, axes = plt.subplots(1, 3 + int(m_inv_b), sharex=True, sharey=True, figsize=figsize);
	axes[0].plot(from_jac);
	axes[0].set_title('from jac')
	axes[1].plot(bias)
	axes[1].set_title('bias')
	axes[2].plot(from_jac.reshape(-1) + bias.reshape(-1))
	axes[2].plot(from_jac.reshape(-1), '--')
	axes[2].set_title('jac + bias')
	if m_inv_b:
		axes[3].plot(-1*np.dot(np.linalg.inv(jac), bias.reshape(-1, 1)))
		axes[3].set_title('-M^{-1}b')
	plt.show()



def plot_filters_of_net(net, layer = 'first'):
	assert(layer in ['first', 'last'])

	if layer == 'first':
		weights_array = net.first_layer.weight.data.numpy()
		weights_array = [x[0] for x in weights_array]
		bias_array = net.first_layer.bias.data.numpy()
	else:
		weights_array = net.last_layer.weight.data[0].numpy()
		bias_array = [net.last_layer.bias.data.numpy()] * len(weights_array)

	for i, x in enumerate(weights_array):
		fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(25, 5));
		axes[0].stem(x)
		axes[0].set_title(str( bias_array[i] ) )
		axes[1].stem(np.abs( np.fft.fftshift( np.fft.fft(x, 
											 n = 30))) )
	plt.show()


def visualize_signal_summary(source,  net):
	
	jac, bias = jacobian_utils.compute_jacobian_and_bias_1d(source, net)
	print('(L2 error between Jacobian and Transpose): ', np.linalg.norm(jac - jac.T), 'norm of jac: ', np.linalg.norm(jac))

	u, s, vt = np.linalg.svd(jac)

	fig, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(20, 5) );
	axes[0].plot(source[0, 0].data.cpu().numpy())
	axes[0].set_title('y_0')
	axes[1].plot(bias)
	axes[1].set_title('bias')
	axes[2].plot(s)
	axes[2].set_title('Singular Values')
	plt.show()
	
	print('Right Singular Vectors')
	plot_singular_vectors(vt.T, k = 5, top = True)
	plt.show()


def visualize_bias_projections(source, noise_sample, net, n_hidden = 5, hidden_dim = 15, kernel_size = 5, only_return = False):
	net_bias = nnet_models.biasproj_DnCNN(n_hidden=n_hidden, 
            hidden_dim = hidden_dim, 
            kernel_size = kernel_size, 
            input_dim = 1)
    
	net_bias.load_state_dict( net.state_dict() );

	input = source + noise_sample;
	source_numpy = source.cpu().data.numpy()[0, 0];

	bias_projs, individual_bias_projs, active_relus = net_bias(input);
	if only_return:
		return bias_projs, individual_bias_projs, active_relus

	n_rows = len(bias_projs);

	fig, axes = plt.subplots(n_rows, 2, sharex=True,  sharey='row', figsize=(20, 20) );
	for i in range(n_rows):
		x_cum, x_ind = bias_projs[i], individual_bias_projs[i]

		axes[i][0].plot(x_cum[0,0])
		axes[i][0].set_title(str(i+1) + ' cum, corr with sig: ' + str(np.corrcoef(np.ravel(source_numpy), np.ravel( x_cum[0,0] ))[0, 1] ) )
		axes[i][1].plot(x_ind[0,0])
		axes[i][1].set_title(str(i+1) + ' ind, corr with sig: ' + str(np.corrcoef(np.ravel(source_numpy), np.ravel( x_ind[0,0] ))[0, 1] ) )


	plt.show()

	# fig, axes = plt.subplots(n_rows, 15, sharex=True,  sharey='row', figsize=(20, 20) );
	# for i in range(n_rows):
	# 	for j in range(15):
	# 		axes[i][j].stem(active_relus[i][0,j].cpu().data.numpy())
	# plt.show()



def visualize_intermediate_jacobians(source, noise_sample = None, net = None, 
										hidden_dim = 15, n_hidden = 5, kernel_size = 5):
	net_intout = nnet_models.intermediate_out_DnCNN(n_hidden = n_hidden, 
									hidden_dim = hidden_dim, 
									kernel_size = kernel_size,
									input_dim = 1);
	net_intout.load_state_dict( net.state_dict() );

	if noise_sample is None:
		input = source
	else:
		input = source + noise_sample;


	for layer in range(n_hidden + 2):
		# print(layer)
		net_temp = partial(net_intout, out_idx = layer);
		jac, bias = jacobian_utils.compute_jacobian_and_bias_1d(input, net_temp);

		if jac.shape[0] == jac.shape[1]:
			print('\nLast Layer')
			plot_rows_of_jac(jac)
		else:	
			assert(jac.shape[0] == jac.shape[1]*hidden_dim)
			print('Layer ', layer+1)
			plt.imhow(jac)
			print('='*50)
			print('\n \n')
	# source_numpy = source.cpu().data.numpy()[0, 0];


def get_subspace_dimension(source, net, noise_std, n_iter = 10, thres = 0.5):

	n_signal = source.shape[-1];
	u_mat = np.zeros([n_signal, 0]);
	v_mat = np.zeros([n_signal, 0]);
	out_mat = np.zeros([n_signal, n_iter]);
	u_rank = np.zeros(n_iter);
	v_rank = np.zeros(n_iter);

	for i in range(n_iter):
		noise_sample = training_utils.get_noise(source, noise_std)
		x_in = source + noise_sample;	
		jac, bias = jacobian_utils.compute_jacobian_and_bias_1d(x_in, net)
		out_mat[:, i] = net(x_in).data.numpy()[0,0].reshape(-1);
		u, s, vt = np.linalg.svd(jac)
		thresh_idx = np.sum( s >= thres );
		u_mat = np.hstack([u_mat, u[:, :thresh_idx]]);
		v_mat = np.hstack([v_mat, vt.T[:, :thresh_idx]]);
		u_rank[i] = np.linalg.matrix_rank(u_mat);
		v_rank[i] = np.linalg.matrix_rank(v_mat);

	plt.plot(u_rank, label = 'u');
	plt.plot(v_rank, label = 'v');
	plt.legend()
	plt.show()

	cov_mat = np.cov(out_mat)
	u, s, vt = np.linalg.svd(cov_mat);
	print('eigenspectrum')
	plot_singular_values(s)
	print('top 5 eig vecs')
	plot_singular_vectors(u, k=5)

	return out_mat


	
def visualize_sing_values_and_vecs(source, net,
									 add_noise = False, noise_std = 0.1, add_blur = False, blur_sigma = 1, blur_kernel_size = 9, 
									 hidden_dim = 15, n_hidden = 5, kernel_size = 5, 
									 use_affine_approx = False, bias = True):



	x_in = source.clone();

	if add_blur:
		x_in = training_utils.blur_with_gaussian(x_in, kernel_size = blur_kernel_size, sigma = blur_sigma);

	if add_noise:
		noise_sample = training_utils.get_noise(x_in, noise_std)
		x_in = x_in + noise_sample;
	
	visualize_denoising(source, net,  x_in = x_in, figsize = (20, 5));
	plt.show()
	
	if use_affine_approx:
		affine_net = affine_approx_around_source.get_affine_approx(source, net, noise_std, bias = bias, 
																		max_epoch = 50, milestone = 40,
																		plot_loss = True, verbose = False, verbose_freq = 10);
		jac = affine_net.linear_layer.weight.cpu().data.numpy()
		bias = affine_net.linear_layer.bias.data.numpy()
		print('With Affine Approximation: ')
		visualize_denoising(source[0], affine_net,  x_in = x_in[0], figsize = (20, 5));
		plt.show()
	else:
		jac, bias = jacobian_utils.compute_jacobian_and_bias_1d(x_in, net)
	print('(L2 error between Jacobian and Transpose): ', np.linalg.norm(jac - jac.T), 'norm of jac: ', np.linalg.norm(jac))

	u, s, vt = np.linalg.svd(jac)

	

	
	print('Bias')
	plt.plot(bias, label = 'bias')
	bias_proj_to_col_space_v = np.dot(vt[:10, :].T, np.dot(vt[:10, :], bias.reshape(-1, 1)) );
	plt.plot(bias_proj_to_col_space_v, label ='bias proj v[:10]')
	plt.plot(np.sum(jac, 1), label='jac row sum');
	plt.title('bias, corr with noise:' + str(np.corrcoef(np.ravel(bias), np.ravel( noise_sample[0,0].cpu().data.numpy() ))[0, 1] ) )
	plt.legend()
	plt.show()


	print('Jac*input + bias = output')
	contribution_from_jac_and_bias(jac, bias, x_in)
	
	
	plot_singular_values(s)
	plt.show()

	if add_noise:
		print('Correlation With U and V')
		plot_corr_with_u_and_v(source, noise_sample, u, vt);	
		plt.show()


	print('Right Singular Vectors')
	plot_singular_vectors(vt.T, k = 5, top = True)
	plot_singular_vectors(vt.T, k = 5, top = False)
	plt.show()
	
	print('Left Singular Vectors')
	plot_singular_vectors(u, k = 5, top = True)
	plot_singular_vectors(u, k = 5, top = False)
	plt.show()
	
	thresh = 1e-5;
	one_sing_idx = np.where( np.abs(s - 1) < thresh)[0];
	all_projection_plots(u, s, vt, x_in[0,0].data.cpu().numpy(), 
						 bias, one_sing_idx, figsize=(20, 5))

	plt.show()


	print('Rows of Jacobian')
	# jac_mat, norm_jac_mat = jacobian_utils.compute_jacobian_norm_jac_grad(x_in, net);
	# plot_rows_of_jac(jac_mat, norm_jac_mat)
	plot_rows_of_jac(jac)



	# print('Projection of Bias')
	# if not add_noise:
	# 	noise_sample = torch.zeros_like(source);
	# visualize_bias_projections(source, noise_sample, net, n_hidden = n_hidden, hidden_dim = hidden_dim, kernel_size = kernel_size)

	# print('Last Layer Acitvations')
	# second_last = net(x_in, return_last = True);
	# second_last = second_last.cpu().data.numpy();

	# n_rows = hidden_dim//3;
	# fig, axes = plt.subplots(n_rows, 3, sharex=True,  sharey='row', figsize=(20, 20) );
	# for i in range(n_rows):
	# 	for j in range(3):
	# 		axes[i][j].stem(second_last[0,i*3+j])
	# plt.show()


