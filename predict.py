import torch
from utils.helpers import *
import warnings
from PIL import Image
from torchvision import transforms
#from torchsummary import summary

def image_transform(imagepath):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image = Image.open(imagepath)
    imagetensor = test_transforms(image)
    return imagetensor


def predict(imagepath, verbose=False):
    if not verbose:
        warnings.filterwarnings('ignore')
    model_path = './results/checkpoint_000001/model.pt'
    try:
        checks_if_model_is_loaded = type(model)
    except:
        model = load_model(model_path)
    model.eval()
    #summary(model, input_size=(3,244,244))
    if verbose:
        print("Model Loaded..")
    image = image_transform(imagepath)
    image1 = image[None,:,:,:]
    ps=torch.exp(model(image1))
    topconf, topclass = ps.topk(1, dim=1)
    if topclass.item() == 1:
        return {'class':'dog','confidence':str(topconf.item())}
    else:
        return {'class':'cat','confidence':str(topconf.item())}

print("Prediction for Dog 1", predict('data/dog1.jpeg'))
print("Prediction for Cat 1", predict('data/cat1.jpeg'))
print("Prediction for Dog 2", predict('data/dog2.jpeg'))
print("Prediction for Cat 2", predict('data/cat2.jpeg'))
print("Prediction for Dog 3", predict('data/dog3.png'))
print("Prediction for Cat 3", predict('data/cat3.png'))
