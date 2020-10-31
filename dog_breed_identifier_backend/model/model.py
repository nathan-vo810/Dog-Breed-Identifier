import torch
from PIL import Image
from torchvision import transforms, models
from torch.autograd import Variable

class Model:
	def __init__(self):
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		return

	def load(self, model_path):
		self.dog_identifier_model = models.wide_resnet50_2()
		num_features = self.dog_identifier_model.fc.in_features
		self.dog_identifier_model.fc = torch.nn.Linear(num_features, 120)

		self.dog_identifier_model.load_state_dict(torch.load(model_path, map_location='cpu'))
		self.dog_identifier_model.eval()

		self.dog_detector_model = models.vgg16(pretrained=True)

	def _load_image(self, image):
		# image = Image.open(image).convert("RGB")

		data_transform = transforms.Compose([transforms.Scale(224),
		                               transforms.CenterCrop(224),
		                               transforms.ToTensor(),
		                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		image = data_transform(image)[:3, :, :].unsqueeze(0)

		return image

	def _detect_dog(self, image):
		output = self.dog_detector_model(image)
		return torch.max(output, 1)[1].item()

	def predict(self, image):
		image = self._load_image(image)
		image = Variable(image).to(self.device)

		if (151 <= self._detect_dog(image) <= 268):
			output = self.dog_identifier_model(image)
			p = torch.nn.functional.softmax(output, dim=1)
			p = p.data.cpu().numpy()

			return p[0], p[0].argsort()[-5:][::-1]

		else:
			print("No dog detected!")
			return None, None