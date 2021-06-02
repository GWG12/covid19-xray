import torch
from PIL import Image
from nn.helpers import get_default_device, to_device
import torchvision.transforms as tt
import os


#model_ft = torch.load(r'D:\\Documents\\Python\\Projects\\fsdl\\app\\nn\\covid19.pth',map_location=torch.device('cpu'))
model_ft = torch.load(r'/app/nn/covid19.pth',map_location=torch.device('cpu'))


tfms = tt.Compose([
    tt.Resize((300,300)),
    tt.ToTensor(),
])


def predict_image(img_path, model):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = tfms(img)
    classes = ['covid19negative', 'covid19positive']
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return classes[preds[0].item()]

device = get_default_device()