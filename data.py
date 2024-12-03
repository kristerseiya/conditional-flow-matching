
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import sys


def get_transform(mode):
    if mode.lower() == 'train':
        transform = transforms.Compose([transforms.RandomVerticalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.ColorJitter(),
                                        transforms.ToTensor(),
                                       ])

    elif mode.lower() in ['val', 'test']:
        transform = transforms.Compose([transforms.ToTensor()])

    elif mode.lower() == 'none':
        transform = transforms.Compose([])

    return transform

class Rescale():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        new_size = (int(x.size[0] * self.scale), int(x.size[1] * self.scale))
        return transforms.Resize(new_size)(x)

class ImageDataSubset(Dataset):
    def __init__(self, dataset, indices, mode='none', patch_size=-1, repeat=1):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.repeat = repeat
        self.patch_size = patch_size
        self.transform = get_transform(mode)
        self.patch_size = patch_size
        if (self.patch_size > 0):
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))
        if (self.dataset.scale > 0) and (self.dataset.store == 'DISK'):
            self.transform.transforms.insert(0, Rescale(self.dataset.scale))

    def set_mode(self, mode):
        self.transform = get_transform(mode)
        if (self.patch_size > 0):
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))
        if (self.dataset.scale > 0) and (self.dataset.store == 'DISK'):
            self.transform.transforms.insert(0, Rescale(self.dataset.scale))

    def set_patch(self, patch_size):
        if (self.patch_size > 0):
            if (self.dataset.scale > 0) and (self.dataset.store == 'DISK'):
                self.transform.transforms.pop(1)
            else:
                self.transform.transforms.pop(0)
        self.patch_size = patch_size
        if (patch_size > 0):
            if (self.dataset.scale > 0) and (self.dataset.store == 'DISK'):
                self.transform.transforms.insert(1, transforms.RandomCrop(patch_size))
            else:
                self.transform.transforms.insert(0, transforms.RandomCrop(patch_size))

    def __getitem__(self, idx):
        if self.dataset.store == 'DISK':
            if self.dataset.gray:
                return self.transform(Image.open(self.dataset.images[self.indices[idx // self.repeat]]).convert('L'))
            else:
                return self.transform(Image.open(self.dataset.images[self.indices[idx // self.repeat]]).convert('RGB'))
        return self.transform(self.dataset.images[self.indices[idx // self.repeat]])

    def __len__(self):
        return len(self.indices) * self.repeat

class ImageDataset(Dataset):
    def __init__(self, root_dirs, mode='none', gray=True, scale=-1,
                 patch_size=-1, repeat=1, store='RAM', extensions='png'):
        super().__init__()
        self.images = list()
        self.store = store.upper()
        self.repeat = repeat
        self.scale = scale
        self.patch_size = patch_size
        self.gray = gray

        if type(extensions) != list:
            extensions = [extensions]

        if type(root_dirs) != list:
            root_dirs = [root_dirs]

        file_paths = list()
        for root_dir in root_dirs:
            for ext in extensions:
                file_paths += glob.glob(os.path.join(root_dir, '*.'+ext))
        n_files = len(file_paths)

        if self.store == 'DISK':
            self.images = file_paths
        elif self.store == 'RAM':
            pbar = tqdm(total=n_files, position=0, leave=False, file=sys.stdout)

            for file_path in file_paths:
                fptr = Image.open(file_path)
                if self.gray:
                    fptr = fptr.convert('L')
                else:
                    fptr = fptr.convert('RGB')
                file_copy = fptr.copy()
                fptr.close()
                if scale > 0:
                    file_copy = Rescale(scale)(file_copy)
                self.images.append(file_copy)
                pbar.update(1)

            tqdm.close(pbar)

        self.transform = get_transform(mode)
        if (self.patch_size > 0):
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))
        if (self.scale > 0) and (self.store == 'DISK'):
            self.transform.transforms.insert(0, Rescale(self.scale))

    def set_mode(self, mode):
        self.transform = get_transform(mode)
        if (self.patch_size > 0):
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))
        if (self.scale > 0) and (self.store == 'DISK'):
            self.transform.transforms.insert(0, Rescale(self.scale))

    def set_patch(self, patch_size):
        if (self.patch_size > 0):
            if (self.scale > 0) and (self.store == 'DISK'):
                self.transform.transforms.pop(1)
            else:
                self.transform.transforms.pop(0)
        self.patch_size = patch_size
        if (patch_size > 0):
            if (self.scale > 0) and (self.store == 'DISK'):
                self.transform.transforms.insert(1, transforms.RandomCrop(patch_size))
            else:
                self.transform.transforms.insert(0, transforms.RandomCrop(patch_size))

    def __len__(self):
        return int(len(self.images) * self.repeat)

    def __getitem__(self, idx):
        if self.store == 'DISK':
            if self.gray:
                return self.transform(Image.open(self.images[idx // self.repeat]).convert('L'))
            else:
                return self.transform(Image.open(self.images[idx // self.repeat]).convert('RGB'))
        return self.transform(self.images[idx // self.repeat])

    def split(self, *r):
        ratios = np.array(r)
        ratios = ratios / ratios.sum()
        total_num = len(self.images)
        indices = np.arange(total_num)
        np.random.shuffle(indices)

        subsets = list()
        start = 0
        for r in ratios[:-1]:
            split = int(total_num * r)
            subsets.append(ImageDataSubset(self, indices[start:start+split]))
            start = start + split
        subsets.append(ImageDataSubset(self, indices[start:]))

        return subsets


class ImageDataset_8_16_32_64(Dataset):
    def __init__(self, root_dirs, mode='none', gray=True, 
                 store='RAM', extensions='png'):
        super().__init__()
        self.img8 = list()
        self.img16 = list()
        self.img32 = list()
        self.img64 = list()
        self.store = store.upper()
        self.gray = gray

        if type(extensions) != list:
            extensions = [extensions]

        if type(root_dirs) != list:
            root_dirs = [root_dirs]

        file_paths = list()
        for root_dir in root_dirs:
            for ext in extensions:
                file_paths += glob.glob(os.path.join(root_dir, '*.'+ext))
        n_files = len(file_paths)

        if self.store == 'DISK':
            raise NotImplementedError()
        elif self.store == 'RAM':
            pbar = tqdm(total=n_files, position=0, leave=False, file=sys.stdout)

            for file_path in file_paths:
                fptr = Image.open(file_path)
                if self.gray:
                    fptr = fptr.convert('L')
                else:
                    fptr = fptr.convert('RGB')
                file_copy = fptr.copy()
                fptr.close()
                img8 = transforms.Resize((8,8))(file_copy)
                img16 = transforms.Resize((16,16))(file_copy)
                img32 = transforms.Resize((32,32))(file_copy)
                self.img8.append(img8)
                self.img16.append(img16)
                self.img32.append(img16)
                self.img64.append(file_copy)
                pbar.update(1)

            tqdm.close(pbar)

        # self.transform = get_transform(mode)
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.scale = 8

    def set_scale(self, scale: int):
        self.scale = scale

    def __len__(self):
        return int(len(self.img8))

    def __getitem__(self, idx):
        if self.scale == 8:
            return self.transform(self.img8[idx])
        elif self.scale == 16:
            return ( self.transform(self.img8[idx]), 
                    self.transform(self.img16[idx]) )
        elif self.scale == 32:
            return ( self.transform(self.img16[idx]), 
                    self.transform(self.img32[idx]) )
        elif self.scale == 64:
            return ( self.transform(self.img32[idx]), 
                     self.transform(self.img64[idx]) )
        else:
            raise RuntimeError()

    def split(self, *r):
        ratios = np.array(r)
        ratios = ratios / ratios.sum()
        total_num = len(self.images)
        indices = np.arange(total_num)
        np.random.shuffle(indices)

        subsets = list()
        start = 0
        for r in ratios[:-1]:
            split = int(total_num * r)
            subsets.append(ImageDataSubset(self, indices[start:start+split]))
            start = start + split
        subsets.append(ImageDataSubset(self, indices[start:]))

        return subsets


# def get_gauss2d(h, w, sigma):
#     gauss_1d_w = np.array([np.exp(-(x-w//2)**2/float(2**sigma**2)) for x in range(w)])
#     gauss_1d_w = gauss_1d_w / gauss_1d_w.sum()
#     gauss_1d_h = np.array([np.exp(-(x-h//2)**2/float(2**sigma**2)) for x in range(h)])
#     gauss_1d_h = gauss_1d_h
#     gauss_2d = np.array([gauss_1d_w * s for s in gauss_1d_h])
#     gauss_2d = gauss_2d / gauss_2d.sum()
#     return gauss_2d

class inputfn:
    def __init__(self, noiselvl, map=False, clip=False):
        # filter = get_gauss2d(5, 5, 2)
        # filter = torch.from_numpy(filter)
        # filter = filter.unsqueeze(0)
        self.noiselvl = noiselvl
        self.map = map
        self.clip = clip

    def __call__(self, x):
        batch_size = x.size(0)
        if type(self.noiselvl) == list:
            sigma = torch.rand(batch_size, 1, 1, 1, device=x.device)
            # sigma = torch.rand_like(images)
            sigma = sigma * (self.noiselvl[1] - self.noiselvl[0]) + self.noiselvl[0]
            # sigma = F.conv2d(sigma, filter, padding='same')
            # sigma = torch.sqrt((self.noiselvl[1] - self.noiselvl[0])**2 * sigma) + self.noiselvl[0]
        noise = torch.randn_like(x) * sigma / 255.
        noisy = x + noise
        if self.clip:
            noisy = torch.clip(noisy, 0, 1)
        if self.map:
            map = sigma.expand_as(noisy) / 255.
            return noisy, map
        return noisy