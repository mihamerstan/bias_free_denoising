import numpy as np
import random
import torch
import torch.utils.data as udata

'''
torch.utils.data.DataLoader takes 2 types of datasets: 
map-style datasets: have a __len__ and __get_item__ protocol
iterable-style datasets: have __iter__() protocol
PieceWiseConstantDataset is a map-style dataset  
'''

class LidarLstmDataset(udata.Dataset):
    def __init__(self, x, y):
        super(LidarLstmDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,index):
        return self.x[index],self.y[index]

 #    def add_missing_pts(self, first_return_df):
	#     # Create a series with the indices of points after gaps and the number of missing points (max of 5)
	#     miss_pt_ser = first_return_df[(first_return_df['miss_pts_before']>0)&\
	#                                       (first_return_df['miss_pts_before']<6)]['miss_pts_before']
	#     # miss_pts_arr is an array of zeros that is the dimensions [num_missing_pts,cols_in_df]
	#     miss_pts_arr = np.zeros([int(miss_pt_ser.sum()),first_return_df.shape[1]])
	#     # Create empty series to collect the indices of the missing points
	#     indices = np.ones(int(miss_pt_ser.sum()))

	#     # Fill in the indices, such that they all slot in in order before the index
	#     i=0
	#     for index, row in zip(miss_pt_ser.index,miss_pt_ser):
	#         new_indices = index + np.arange(row)/row-1+.01
	#         indices[i:i+int(row)] = new_indices
	#         i+=int(row)
	#     # Create a Dataframe of the indices and miss_pts_arr
	#     miss_pts_df = pd.DataFrame(miss_pts_arr,index=indices,columns = first_return_df.columns)
	#     miss_pts_df['mask'] = [0]*miss_pts_df.shape[0]
	#     # Fill scan fields with NaN so we can interpolate them
	#     for col in ['scan_angle','scan_angle_deg']:
	#         miss_pts_df[col] = [np.NaN]*miss_pts_df.shape[0]
	#     # Concatenate first_return_df and new df
	#     full_df = first_return_df.append(miss_pts_df, ignore_index=False)
	#     # Resort so that the missing points are interspersed, and then reset the index
	#     full_df = full_df.sort_index().reset_index(drop=True)
	#     return full_df

	# def min_max_tensor(self,tensor):
	#     # Function takes a 3-D tensor, performs minmax scaling to [0,1] along the third dimension.
	#     # First 2 dimensions are flattened
	#     a,b,c = tensor.shape
	#     # Flatten first two dimensions
	#     flat_tensor = tensor.view(-1,c)
	#     sc =  MinMaxScaler()
	#     flat_norm_tensor = sc.fit_transform(flat_tensor)
	#     # Reshape to original
	#     output = flat_norm_tensor.reshape([a,b,c])
	#     return torch.Tensor(output), sc

	# def sliding_windows(self, data, seq_length, line_num, x, y):
	#     for i in range(len(data)-seq_length):
	#         # Index considers previous lines
	#         idx = i+line_num*(min_pt_count-seq_length)
	#         _x = data[i:(i+seq_length)]
	#         _y = data[i+seq_length,:3] # Assumes xyz are the first 3 features in scan_line_tensor
	#         x[idx,:,:] = _x
	#         y[idx,:,:] = _y

	#     return x,y

	# def generate_samples(self,data,min_pt_count,seq_len,num_scan_lines,val_split,starting_line=1000):
	#     '''
	#     Function generates training and validation samples for predicting the next point in the sequence.
	#     Inputs:
	#         data: 3-Tensor with dimensions: i) the number of viable scan lines in the flight pass, 
	#                                         ii) the minimum number of points in the scan line,
	#                                         iii) 3 (xyz, or feature count)
	    
	#     '''
	#     # Create generic x and y tensors
	#     x = torch.ones([(min_pt_count-seq_len)*num_scan_lines,seq_len,len(feature_list)]) 
	#     y = torch.ones([(min_pt_count-seq_len)*num_scan_lines,1,3])
	#     i=0
	#     # Cycle through the number of scan lines requested, starting somewhere in the middle
	#     for line_idx in range(starting_line,starting_line+num_scan_lines):
	#         x,y = sliding_windows(data[line_idx,:,:],seq_len,line_idx-starting_line, x, y)
	#     x_train,y_train,x_val,y_val = train_val_split(x,y,val_split)
	#     return x_train,y_train,x_val,y_val

	# def train_val_split(self,x,y,val_split):   
	#     # Training/Validation split
	#     # For now, we'll do the last part of the dataset as validation...shouldn't matter?
	#     train_val_split_idx = int(x.shape[0]*(1-val_split))
	#     x_train = x[:train_val_split_idx,:,:]
	#     x_val = x[train_val_split_idx:,:,:]
	#     y_train = y[:train_val_split_idx,:,:]
	#     y_val = y[train_val_split_idx:,:,:]
	    
	#     return x_train,y_train,x_val,y_val

 #    def get_lidar(self):
 #    	first_return_df = pd.read_pickle("../../../Data/parking_lot/first_returns_modified_164239.pkl")
 #    	# Note: x_scaled, y_scaled, and z_scaled MUST be the first 3 features
	# 	feature_list = [
	# 	    'x_scaled',
	# 	    'y_scaled',
	# 	    'z_scaled',
	# 	    'scan_line_idx',
	# 	    'scan_angle_deg',
	# 	    'abs_scan_angle_deg'
	# 	]
	# 	# miss_pts_before is the count of missing points before the point in question (scan gap / 5 -1)
	# 	first_return_df['miss_pts_before'] = round((first_return_df['scan_gap']/-5)-1)
	# 	first_return_df['miss_pts_before'] = [max(0,pt) for pt in first_return_df['miss_pts_before']]

	# 	# Add 'mask' column, set to one by default
	# 	first_return_df['mask'] = [1]*first_return_df.shape[0]

	# 	first_return_df = self.add_missing_pts(first_return_df)
	# 	first_return_df[['scan_angle','scan_angle_deg']] = first_return_df[['scan_angle','scan_angle_deg']].interpolate()
	# 	first_return_df['abs_scan_angle_deg'] = abs(first_return_df['scan_angle_deg'])

	# 	# Extract tensor of scan lines
	# 	# Number of points per scan line
	# 	scan_line_pt_count = first_return_df.groupby('scan_line_idx').count()['gps_time']

	# 	# Identify the indices for points at end of scan lines
	# 	scan_break_idx = first_return_df[(first_return_df['scan_gap']>scan_line_gap_break)].index

	# 	# Create Tensor
	# 	line_count = ((scan_line_pt_count>min_pt_count)&(scan_line_pt_count<max_pt_count)).sum()
	# 	scan_line_tensor = torch.randn([line_count,min_pt_count,len(feature_list)])

	# 	# Collect the scan lines longer than min_pt_count
	# 	# For each, collect the first min_pt_count points
	# 	i=0
	# 	for line,count in enumerate(scan_line_pt_count):
	# 	    if (count>min_pt_count)&(count<max_pt_count):
	# 	        try:
	# 	            line_idx = scan_break_idx[line-1]
	# 	            scan_line_tensor[i,:,:] = torch.Tensor(first_return_df.iloc\
	# 	                                      [line_idx:line_idx+min_pt_count][feature_list].values)
	# 	            i+=1
	# 	        except RuntimeError:
	# 	            print("line: ",line)
	# 	            print("line_idx: ",line_idx)

	# 	scan_line_tensor_norm, sc = self.min_max_tensor(scan_line_tensor)
