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

class PieceWiseConstantDataset(udata.Dataset):
    '''
    n_data: number of examples
    n_signal: number of samples per example
    prob: probability of the signal changing value
    fix_datapoints: if True, data is determined and fixed. if False, data is generated when __get_item__ is called.
    min_val: min signal value
    max_val: max signal value
    min_sep: higher min_sep means the signal is more likely to remain flat
    '''
    def __init__(self, n_data = 1000, n_signal = 64, prob=0.3, fix_datapoints = True,
                  min_val = 0, max_val = 1, min_sep = 5):
        super(PieceWiseConstantDataset, self).__init__()

        assert(min_sep > 0)
        self.n_data = n_data;
        self.n_signal = n_signal;
        self.prob = prob
        self.fix_datapoints = fix_datapoints;
        self.min_val = min_val;
        self.max_val = max_val;
        self.min_sep = min_sep;
        
        # This fixes the points so that each time __get_item__ is called, the same data is returned.
        if self.fix_datapoints:
            self.data_list = [None] * self.n_data;
            for i in range(self.n_data):
                # self.data_list[i] = self.gen_piecewise_constant(self.n_signal, self.prob, min_val=self.min_val, max_val=self.max_val);
                self.data_list[i] = self.gen_piecewise_constant_minsep(self.n_signal, self.prob, self.min_sep,
                                                    min_val=self.min_val, max_val=self.max_val);

    def __len__(self):
        return self.n_data

    # # Not currently active, superseded by gen_piecewise_constant_minsep()
    # def gen_piecewise_constant(self, n, prob, min_val, max_val):
    #    signal = np.zeros(n)
    #    val = np.random.uniform(min_val, max_val);
    #    ind = np.random.randint(n)
    #    for i in range(n):
    #        signal[ind] = val
    #        if np.random.rand() < prob:
    #            val = np.random.uniform(min_val, max_val)
    #        ind = np.mod(ind + 1,n)
    #    return signal

    def gen_piecewise_constant_minsep(self, n, prob, minsep, min_val, max_val):
        np.random.seed() # MS: Added this, otherwise numpy generates the same random data each time.
        signal = np.zeros(n)
        ind = 0
        # val = np.random.uniform(min_val, max_val)
        val = torch.rand(1)*(max_val-min_val)+min_val
        while ind < n:
            if ind + minsep > n:
                signal[ind:] = val
                break
            if ind == 0 or torch.rand(1) < prob:
                # val = np.random.uniform(min_val, max_val)
                val = torch.rand(1)*(max_val-min_val)+min_val
                # sep = np.random.randint(1, minsep+1) if ind == 0 else minsep # always =1 if minsep=1
                
                # This line allows shorter pieces at the front of the signal - removing for now
                # sep = torch.randint(1, minsep+1,(1,)) if ind == 0 else minsep # always =1 if minsep=1
                sep = minsep
                if ind + sep > n:
                    signal[ind:] = val
                    break
                else:
                    signal[ind:(ind+sep)] = val
                    ind += sep
            else:
                signal[ind] = val
                ind += 1
        return signal

    def __getitem__(self, index):
        
        if self.fix_datapoints:
            signal = self.data_list[index]
        else:
            # signal = self.gen_piecewise_constant(self.n_signal, self.prob, min_val=self.min_val, max_val=self.max_val);
            signal = self.gen_piecewise_constant_minsep(self.n_signal, self.prob, self.min_sep,
                                                    min_val=self.min_val, max_val=self.max_val)
        return torch.from_numpy(signal).unsqueeze(0).type(torch.FloatTensor)


class MaskedDataset(udata.Dataset):
    '''
    Returns the PieceWiseConstantDataset signal as 'clean', mask binary mapping as 'mask', and masked signal as 'masked'
    
    PieceWiseConstantDataset params:
    n_data: number of examples
    n_signal: number of samples per example
    prob: probability of the signal changing value
    fix_datapoints: if True, data is determined and fixed. if False, data is generated when __get_item__ is called.
    min_val: min signal value
    max_val: max signal value
    min_sep: higher min_sep means the signal is more likely to remain flat

    Mask params:
    mask_length: the number of samples that are masked in succession
    '''
    def __init__(self,  n_data = 1000, n_signal = 64, prob=0.3, fix_datapoints = True,
                  min_val = 0, max_val = 1, min_sep = 5, mask_length_min = 5, mask_length_max = 10,
                  test_num = 0):
        super(MaskedDataset, self).__init__()
        

        self.PieceWiseConstantDataset = PieceWiseConstantDataset(n_data, n_signal, prob, fix_datapoints,
                                                                    min_val, max_val, min_sep) 
        # TO DO: Init any mask parameters
        # self.mode = mode
        # self.noise_std = noise_std
        # self.max_noise = max_noise
        self.mask_length_min = mask_length_min
        self.mask_length_max = mask_length_max
        self.test_num = test_num

    def get_mask(self, data, test_num=0):
        mask_length = int(torch.randint(self.mask_length_min,self.mask_length_max+1,(1,)))
        mask = torch.ones(data.shape)
        # Select the starting point of the mask
        if test_num == 0:
            mask_start = torch.randint(0,mask.shape[1]-mask_length+1,(1,))
            mask[0][mask_start:mask_start+mask_length] = 0
        elif test_num == 1:
            # Test 1: Mask at beginning of signal
            mask_start = 0
            # Setting small mask length for now
            mask[0][mask_start:mask_start+5] = 0
            # mask[0][mask_start:mask_start+mask_length] = 0
        elif test_num == 2:
            # Test 2: Mask in middle of signal
            mask_start = 20
            mask[0][mask_start:mask_start+mask_length] = 0
        elif test_num == 3:
        #     # Test 3: Only 2 constant values before and after the signal 
        #     mask_start = torch.randint(2,mask.shape[1]-mask_length+1,(1,))
        #     mask[0][mask_start:mask_start+mask_length] = 0
            mask = torch.Tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
              1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1.,0.,0.,0.,0.,0.,0.,0.,0., 
              0., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        else:
            return "An invalid test_num was provided to MaskedDataset class."
        return mask
    
    def __len__(self):
        return len(self.PieceWiseConstantDataset)
    
    def __getitem__(self, index):     
        if self.test_num == 3:
            clean = torch.Tensor([[0.7190, 0.7190, 0.7190, 0.7190, 0.7190, 0.7190, 0.7190,
                  0.8647, 0.8647,0.8647, 0.8647, 0.8647, 0.8647, 0.8647, 0.8647, 
                  0.9960, 0.9960,0.9960, 0.9960, 0.9960, 
                  0.4230,0.4230,0.4230,0.4230,0.4230,0.4230,0.4230,
                  0.9101,0.9101,0.9101, 0.9101,0.9101,0.9101, 
                  0.0530, 0.0530, 0.0530,0.0530, 0.0530, 
                  0.4974,0.4974, 0.4974, 0.4974, 0.4974, 0.4974,
                  0.1920, 0.1920, 0.1920, 0.1920,0.1920, 0.1920]])
            mask = self.get_mask(clean,self.test_num)
        else:
            clean = self.PieceWiseConstantDataset[index]
            mask = self.get_mask(clean,self.test_num)
        return clean, mask