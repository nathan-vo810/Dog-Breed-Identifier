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
	dog_identifier_model = models.wide_resnet50_2()
	num_features = dog_identifier_model.fc.in_features
	dog_identifier_model.fc = torch.nn.Linear(num_features, 120)

	model_path = "./wide_resnet_883562.pth"
	dog_identifier_model.load_state_dict(torch.load(model_path, map_location='cpu'))
	dog_identifier_model.eval()

	dog_detector_model = models.vgg16(pretrained=True)

	return dog_detector_model, dog_identifier_model

def load_image(image_path):
	image = Image.open(image_path).convert("RGB")

	data_transform = transforms.Compose([transforms.Scale(224),
	                               transforms.CenterCrop(224),
	                               transforms.ToTensor(),
	                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	image = data_transform(image)[:3, :, :].unsqueeze(0)

	return image

def detect_dog(model, image):
	output = model(image)
	return torch.max(output, 1)[1].item()


def predict_image(model, image):	
	output = model(image)
	p = torch.nn.functional.softmax(output, dim=1)
	p = p.data.cpu().numpy()

	return p[0], p[0].argsort()[-5:][::-1]

def main():
	dog_detector_model, dog_identifier_model = prepare_model()
	class_names = get_class_names()

	image_path = './dataset/test/0a3f1f6f5f0ede7ea6e27427994d5f62.jpg'
	image = load_image(image_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	image = Variable(image).to(device)

	if (151 <= detect_dog(dog_detector_model, image) <= 268):
		probabilities, top5_indexes = predict_image(dog_identifier_model, image)

		for i, index in enumerate(top5_indexes):
			print("{} - {}, Confident: {}".format(i+1, class_names[index], probabilities[index]))

	else:
		print("No dog detected!")

if __name__ == '__main__':
	main()