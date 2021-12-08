import numpy as np
from PIL import Image
import os
import sys

import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as TF
from datasetUtil.read_minst import load_mnist_train, load_mnist_test, fiter_mnist

class Dataset(torch.utils.data.Dataset):
	def __init__(self, image_dir, normal_image_dir, dataset_dir=None, dataset_type='train', patch_size=299, fold_list=None, num_instances=10):
		self._num_instances = num_instances
		self._patch_size = patch_size
		self._dataset_dir = dataset_dir
		self._dataset_type = dataset_type 
		self._fold_list = fold_list 
		self._image_dir = image_dir
		self._normal_image_dir = normal_image_dir
		self.target_num = [0, 7]
		self.train_images_filter, self.train_labels_filter = self.read_minst_data()

		# self._num_patients = self._patient_ids_arr.shape[0]
		#
		# self._indices = np.arange(self._num_patients)
		#
		# self._img_transforms = self.image_transforms()


	# @property
	# def num_patients(self):
	# 	return self._num_patients

	def __len__(self):
		return len(self.train_images_filter)

	def read_minst_data(self,  ):
		if self._dataset_type == 'train':
			train_images, train_labels = load_mnist_train('dataset/train', kind='train')
			images_filter, labels_filter = fiter_mnist(train_images, train_labels,  )
		else:
			test_images, test_labels = load_mnist_test('dataset/test', kind='t10k')
			images_filter, labels_filter = fiter_mnist(test_images, test_labels,  )

		return images_filter, labels_filter

	def image_transforms(self):
		if self._dataset_type == 'train':
			img_transformsimg_transforms = transforms.Compose([
													transforms.RandomCrop(self._patch_size),
													transforms.RandomHorizontalFlip(),
													transforms.RandomVerticalFlip(),
													transforms.ToTensor(),
													self.MyNormalizationTransform(),
													])

		else:
			img_transforms = transforms.Compose([
													transforms.ToTensor(),
													self.MyNormalizationTransform(),
													])

		return img_transforms


	class MyNormalizationTransform(object):
		def __call__(self, input_tensor):
			mean_tensor = torch.mean(input_tensor).view((1,))#, dim=(0,1,2))
			std_tensor = torch.std(input_tensor).view((1,))#, dim=(0,1,2))

			if 0 in std_tensor:
				std_tensor[0] = 1.0

			return TF.normalize(input_tensor, mean_tensor, std_tensor)

	def get_sample_data(self, img_dir, patch_ids):
		
		img_tensor_list = list()
		for i in range(len(patch_ids)):
			img_path = '{}/{}.jpeg'.format(img_dir, patch_ids[i])

			img = Image.open(img_path).convert("RGB")

			img_tensor = self._img_transforms(img)

			img_tensor_list.append(img_tensor)

		return torch.stack(img_tensor_list,dim=0)


	def __getitem__(self, idx):

		# temp_index = self._indices[idx]
		#
		# temp_patient_id = self._patient_ids_arr[temp_index]
		# temp_num_patches = self._num_patches_arr[temp_index]
		# temp_label = self._labels_arr[temp_index]
		# temp_img_dir = self._img_dirs_arr[temp_index]
		#
		# patch_indices = np.arange(temp_num_patches)
		#
		# if self._num_instances > temp_num_patches:
		# 	patch_indices = np.repeat(patch_indices, int(self._num_instances//temp_num_patches + 1) )
		#
		# np.random.shuffle(patch_indices)
		# patch_indices = patch_indices[:self._num_instances]

		# temp_sample = self.get_sample_data(img_dir = temp_img_dir, patch_ids = patch_indices)

		train_images_filter = self.train_images_filter[idx]
		train_labels_filter = self.train_labels_filter[idx]
		train_images_filter = torch.as_tensor(train_images_filter, dtype=torch.float32)
		train_labels_filter = torch.as_tensor(train_labels_filter, dtype=torch.float32)
		# temp_label = torch.as_tensor([idx], dtype=torch.float32)

		return train_images_filter, train_labels_filter


def custom_collate_fn(batch):
	sample_tensors_list, label_tensors_list = zip(*batch)
	return torch.cat(sample_tensors_list,dim=0), torch.stack(label_tensors_list,dim=0)

def worker_init_fn(id):
	np.random.seed(torch.initial_seed()&0xffffffff)
