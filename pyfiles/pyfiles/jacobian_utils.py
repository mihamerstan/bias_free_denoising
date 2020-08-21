from torch.autograd.gradcheck import zero_gradients
import torch

def compute_jacobian_and_bias_1d(inputs, net):

    inputs.requires_grad = True
    assert inputs.requires_grad
    
    outputs = net(inputs);

    assert(outputs.dim() == 3)
    assert(outputs.shape[0] == 1)
    assert(inputs.shape[1] == 1)

    channels = outputs.shape[1]
    n = inputs.shape[-1];

    total_dim = n*channels;
    
    jacobian = torch.zeros([total_dim, n]);
    grad_output = torch.zeros(n);
    
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for j in range(channels):
        for i in range(n):
            zero_gradients(inputs)
            outputs[0, j, i].backward(retain_graph = True)
            jacobian[ j*channels + i, :] = inputs.grad.data.view(-1)
            
    
    bias = outputs[0].view(-1) - torch.matmul(jacobian, inputs[0,0])
    
    return jacobian.cpu().numpy(), bias.data.cpu().numpy()