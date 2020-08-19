import os

from PIL import Image
import pickle
import torch.utils.data as data


class AnomalyDataset(data.Dataset):
    def __init__(self, root, split='train', in_channel=3, transform=None, target_transform=None, cls=None):
        """
        Args:
            root (str): Directory that contains splited datasets
            split (list or str): Type of split to load (e.g. 'train', 'val')
            in_channel (int): Number of input channel (e.g. 1 for grayscale data, 3 for RGB data)
            transform (class): Transform applied to the input
            target_transform (class): Transform applied to the target image of reconstruction
            cls (int): An inlier class
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set
        self.in_channel = in_channel
        self.cls = cls
        self.label_img_data = []

        with open(os.path.join(self.root, 'data_split_%s.pkl' % split), 'rb') as pkl:
            split_data = pickle.load(pkl)

        if split.split('_')[0] == 'test':
            self.label_img_data = split_data
        else:
            # Load inlier class samples
            for x in split_data:
                if x[0] == self.cls:
                    self.label_img_data.append(x)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img (tensor): Input image after transform
            target (tensor): Target image after transform for the reconstruction
            label (tensor): Class label for input image
        """
        label, img = self.label_img_data[index][0], self.label_img_data[index][1]

        if self.in_channel == 1:
            img = Image.fromarray(img, mode='L')
        elif self.in_channel == 3:
            img = Image.fromarray(img, mode='RGB')
        target = img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, label

    def __len__(self):
        return len(self.label_img_data)
