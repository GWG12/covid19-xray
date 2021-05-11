import torch


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
    
def to_device(data, device):
    #In you want to pass multiple tensors at the same time: data = [tensor1,tensor2...batch1, batch2....]
    if isinstance (data, (list,tuple)):
        return [to_device(x,device) for x in data]
    # This line is what makes the magic
    return data.to(device, non_blocking=True)