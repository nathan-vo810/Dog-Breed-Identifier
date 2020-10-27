import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models
import copy
import argparse

def prepare_data(data_dir, batch_size):
	data_transform = transforms.Compose([transforms.Scale(224),
	                               transforms.CenterCrop(224),
	                               transforms.ToTensor(),
	                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	image_datasets = {x: datasets.ImageFolder(join(data_dir, x), data_transform) for x in ['train', 'valid']}

	dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=4) for x in ['train', 'valid']}

	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
	class_names = image_datasets['train'].classes

	return dataloaders, dataset_sizes, class_names


def train_model(device, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()    # Set model to training mode
            else:
                model.eval()     # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterates over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameters gradient
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_args():
	# # Create parser object
    parser = argparse.ArgumentParser(description="CLI for Dog Breed Classifier")

    parser.add_argument('-model', type=str, nargs=1,
                        metavar="model", default='ResNet18', choices={'ResNet18', 'ResNet50', 'Wide_ResNet', 'Inception_v3'},
                        help='Choose model type - ResNet18/ResNet50/Wide_ResNet/Inception_v3')

    parser.add_argument('-batch_size', type=int, metavar="batch_size", default=4)

    parser.add_argument('-num_epochs', type=int, metavar="num_epochs", default=1)

    parser.add_argument('-lr', type=float, metavar="learning_rate", default=0.001)

    args = parser.parse_args()
    return args


def main():
	# Get arguments
	args = get_args()

	# Generate random seed
	np.random.seed(0)

	# Prepare device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Prepare Data
	data_dir = './dataset'
	dataloaders, dataset_sizes, class_names = prepare_data(data_dir=data_dir, batch_size=args.batch_size)

	# Prepare Model
	print("Using Model", args.model)
	if args.model == 'ResNet18':
		model = models.resnet18(pretrained=True)
	elif args.model == 'ResNet50':	
		model = models.resnet50(pretrained=True)
	elif args.model == 'Inception_v3':
		model = models.inception_v3(pretrained=True)
	else:
		model = models.wide_resnet50_2(pretrained=True)

	for param in model.parameters():
	    param.requires_grad = False

	num_features = model.fc.in_features

	model.fc = torch.nn.Linear(num_features, len(class_names))

	model = model.to(device)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	# Train Model
	my_model = train_model(device, model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=args.num_epochs)

	# Save Model
	torch.save(my_model.state_dict(), "./best_weights.pth")


if __name__ == '__main__':
	main()