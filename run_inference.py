import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torch.autograd import Variable

def get_class_names():
	with open("class_names.npy", "rb") as f:
		class_names = np.load(f)
	return class_names

def prepare_model():
	model = models.wide_resnet50_2()
	num_features = model.fc.in_features
	model.fc = torch.nn.Linear(num_features, 120)

	model_path = "./wide_resnet_883562.pth"
	model.load_state_dict(torch.load(model_path, map_location='cpu'))
	model.eval()
	return model

def predict_image(model, image_name):
	image = Image.open(image_name)

	data_transform = transforms.Compose([transforms.Scale(224),
	                               transforms.CenterCrop(224),
	                               transforms.ToTensor(),
	                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	image_tensor = data_transform(image).float()
	image_tensor = image_tensor.unsqueeze_(0)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	input = Variable(image_tensor)
	input = input.to(device)

	output = model(input)
	p = torch.nn.functional.softmax(output, dim=1)
	p = p.data.cpu().numpy()

	return p[0], p[0].argsort()[-5:][::-1]

def main():
	model = prepare_model()
	class_names = get_class_names()

	image_name = './dataset/test/0a3f1f6f5f0ede7ea6e27427994d5f62.jpg'
	probabilities, top5_indexes = predict_image(model, image_name)

	for i, index in enumerate(top5_indexes):
		print("{} - {}, Confident: {}".format(i+1, class_names[index], probabilities[index]))


if __name__ == '__main__':
	main()