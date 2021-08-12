import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import sys, os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['TORCH_HOME'] = '/datadrive/model'
class FeatureExtractor(nn.Module):
    def __init__(self,in_channels=3):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet18(pretrained=True)

        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')

        # Set model to evaluation mode
        self.model.eval()

        self.head = nn.Linear(512, 128)


    def forward(self, encoder_inputs, target=False):
        self.model.eval()
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        if encoder_inputs.image_inputs is not None and encoder_inputs.target_image is not None:
            x = encoder_inputs.target_image if target else encoder_inputs.image_inputs
        else:
            print("Error: Please check if the image input is valid and try again.")
            sys.exit(0)
            # x= encoder_inputs.image_inputs if 
        x = x.to(device)
        
        my_embedding = torch.zeros(512).to(device)

        # Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.flatten())                 # <-- flatten

        # Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)
        # Run the model on our transformed image
        with torch.no_grad():                               # <-- no_grad context
            self.model(x)                       # <-- unsqueeze
        # Detach our copy function from the layer
        h.remove()
        # Return the feature vector
        my_embedding = self.head(my_embedding.unsqueeze(0))
        return my_embedding


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


