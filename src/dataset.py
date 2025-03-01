from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import torchvision.transforms.functional as F
import cv2    
torch.multiprocessing.set_sharing_strategy('file_system')

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

class ToChannelsLast:
    def __call__(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim !=4:
            raise RuntimeError
        return x.to(memory_format=torch.channels_last)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_med_transforms(size, mean_train=None, std_train=None):
    data_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(0.5)
        ])
    data_test_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((size, size))])
    gt_test_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((size, size))])
    return data_transforms, data_test_transforms, gt_test_transforms

'''
def get_data_transforms2(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size))])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size))])
    return data_transforms, gt_transforms

def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    return data_transforms
'''

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'CAPS_IXI/png')
            self.gt_path = None
        else:
            self.img_path = os.path.join(root, 'ATLAS/png_sl_95/images')
            self.gt_path = os.path.join(root, 'ATLAS/png_sl_95/masks')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []

        if not self.gt_path:
            img_paths = glob.glob(os.path.join(self.img_path) + "/*.png")
            img_tot_paths.extend(img_paths)
            #img_paths2 = glob.glob('/home/marco/nas_data/Datasets/Anomaly/brain/AOMIC/PIOP2_png_sl_110/*.png')
            #img_tot_paths.extend(img_paths2)
            #img_paths3 = glob.glob('/home/marco/nas_data/Datasets/Anomaly/brain/AOMIC/PIOP1_png_sl_110/*.png')
            #img_tot_paths.extend(img_paths3)
            gt_tot_paths.extend([0] * (len(img_tot_paths)))
            tot_labels.extend([0] * (len(img_tot_paths)))
        else:
            img_paths = glob.glob(os.path.join(self.img_path) + "/*.png")
            gt_paths = glob.glob(os.path.join(self.gt_path) + "/*.png")
            img_paths.sort()
            gt_paths.sort()
            img_tot_paths.extend(img_paths)
            gt_tot_paths.extend(gt_paths)
            tot_labels.extend([1] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    #def __getitem__(self, idx):
    #    img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
    #    img = Image.open(img_path).convert('RGB')
    #    img = self.transform(img)
    #    img = torch.from_numpy(np.array(img))
    #    return img

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if 'Atlas' in img_path or 'AOMIC'in img_path:
            img = img.rotate(90)
            img = np.array(img)
            img = img[:, :, ::-1].copy()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)

            kernel = np.ones((10, 10), np.uint8)
            img_open = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(img_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

            for i in range(3):
                img[:,:,i] = img[:,:,i]*(binary/255)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        
        img = self.transform(img)
        img = torch.from_numpy(np.array(img))
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open((img_path[:-4]+'_mask.png').replace('images', 'masks')).convert('L').rotate(90)
            gt = self.gt_transform(gt)
            gt = torch.from_numpy(np.array(gt)).unsqueeze(0)/255
        #print(img.shape, gt.shape)
        #assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        self.transform = transform
        self.phase = phase
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.from_numpy(np.array(img))
        return img, label, img_path
    
class MedicalDatasetSeg(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, return_masks):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test', 'test')
            self.gt_path = os.path.join(root, 'test', 'test_labels')
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        self.return_masks = return_masks
        # load dataset
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []
        gt_tot_paths = []
        defect_types = os.listdir(self.img_path)
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_paths.sort()
                if self.return_masks:
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*")
                    gt_paths.sort()
                else:
                    gt_paths = [0] * len(img_paths)
                tot_labels.extend([1] * len(img_paths))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)

            assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
            
        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt_path, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('L').convert('RGB')
        img = self.transform(img)
        img = torch.from_numpy(np.array(img))
        if self.return_masks:
            if gt_path == 0:
                gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
            else:
                gt = Image.open(gt_path).convert('L')
                
                # Binarize labels
                if 'RESC' in self.gt_path:
                    gt = (np.array(gt))
                    #gt[gt == 255] = 0 # Only liquid
                    gt[gt > 0] = 255
                    gt = Image.fromarray(gt)
                elif 'BraTS' in self.gt_path:
                    gt = (np.array(gt))
                    gt[gt > 0] = 255
                    gt = Image.fromarray(gt)
                gt = self.gt_transform(gt)
                gt = torch.from_numpy(np.array(gt)).unsqueeze(0)/255
            return img, gt, label, img_path
        else:
            return img, 0, label, img_path


def load_data(device, _class_, size):
        image_size = size
        crop_size = 128
        print(size)
        data_transform, test_transform, gt_transform = get_med_transforms(image_size, crop_size)
        if _class_ == 'OCT':
            use_mask = False
            train_path = '/home/marco/nas_data/Datasets/Anomaly/OCT2017'
            train_data = MedicalDataset(root=train_path, transform=data_transform, gt_transform=None, phase="train")
            test_data = MedicalDataset(root=train_path, transform=test_transform, gt_transform=gt_transform, phase="test")
        elif _class_ == 'RESC':
            train_path = '/home/marco/nas_data/Datasets/Anomaly/RESC'
            use_mask = True
            train_data = MedicalDatasetSeg(root=train_path, transform=data_transform, gt_transform=None, phase="train", return_masks=False)
            test_data = MedicalDatasetSeg(root=train_path, transform=test_transform, gt_transform=gt_transform, phase="test", return_masks=use_mask)
        elif _class_ == 'CHEST':
            train_path = '/home/marco/nas_data/Datasets/Anomaly/Chest-RSNA'
            use_mask = False
            train_data = MedicalDatasetSeg(root=train_path, transform=data_transform, gt_transform=None, phase="train", return_masks=False)
            test_data = MedicalDatasetSeg(root=train_path, transform=test_transform, gt_transform=gt_transform, phase="test", return_masks=use_mask)
        elif _class_ == 'BRAIN':
            train_path = '/home/marco/nas_data/Datasets/Anomaly/BraTS2021_slice'
            use_mask = True
            train_data = MedicalDatasetSeg(root=train_path, transform=data_transform, gt_transform=None, phase="train", return_masks=False)
            test_data = MedicalDatasetSeg(root=train_path, transform=test_transform, gt_transform=gt_transform, phase="test", return_masks=use_mask)
        elif _class_ == 'LIVER':
            train_path = '/home/marco/nas_data/Datasets/Anomaly/hist_DIY'
            use_mask = True
            train_data = MedicalDatasetSeg(root=train_path, transform=data_transform, gt_transform=None, phase="train", return_masks=False)
            test_data = MedicalDatasetSeg(root=train_path, transform=test_transform, gt_transform=gt_transform, phase="test", return_masks=use_mask)
        elif _class_ == 'HIST':
            train_path = '/home/marco/nas_data/Datasets/Anomaly/camelyon16_256'
            use_mask = False
            train_data = MedicalDatasetSeg(root=train_path, transform=data_transform, gt_transform=None, phase="train", return_masks=False)
            test_data = MedicalDatasetSeg(root=train_path, transform=test_transform, gt_transform=gt_transform, phase="test", return_masks=use_mask)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4,
                                                   drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

        return train_dataloader, test_dataloader, use_mask