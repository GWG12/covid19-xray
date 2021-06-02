import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os
from typing import Callable, List


class ImageSimilarity:

    '''
    App: Compares how similar are two images by using the cosine similarity method
    Args: -    
    '''

    def __init__(self):
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.vec_folder = r'D:\\Documents\\Python\\Projects\\fsdl\\app\\image_similarity\\Vectors'
        # Load the pretrained model
        self.model = models.resnet18(pretrained=True)
        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')
        self.treshold = 0.77

    def _load_features_vectors(self, source_folder: str) -> List[torch.tensor]:
        '''
         Loads all pre-saved vectors
        '''
        files = os.listdir(source_folder)
        vectors = []
        for file in files:
            loaded_tensor = torch.load(f'{source_folder}/{file}')
            vectors.append(loaded_tensor)

        return vectors

    def _get_vector(self, image_name) -> torch.tensor:
        '''
        Transforms image into torch tensor
        '''
        # 1. Load the image with Pillow library
        img = Image.open(image_name)
        img = img.convert("L")

        # 2. Create a PyTorch Variable with the transformed image
        #t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
        pre_img = Variable(self.to_tensor(self.scaler(img)))
        pre_img = pre_img.repeat(3, 1, 1)
        t_img = Variable(self.normalize(pre_img).unsqueeze(0))

        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(1, 512, 1, 1)
        # 4. Define a function that will copy the output of a layer

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        self.model(t_img)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        return my_embedding

    def compare_image(self, image_path) -> bool:
        '''
        Applies cosine similarity to image vector and compares it with 100 pre-saved image vectors from
        Covid-19 chest x rays dataset
        '''
        base_vectors = self._load_features_vectors(self.vec_folder)
        print(f'Base vectors length: {len(base_vectors)}')
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sample_img = self._get_vector(image_path)
        acc = 0
        for vector in base_vectors:
            vector_comp = cos(sample_img, vector).item()
            acc += vector_comp
        similarity_mean = acc/len(base_vectors)
        # If more than treshold, it's similar
        if similarity_mean >= self.treshold:
            return True
        return False
