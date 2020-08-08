import os
from PIL import Image
from torchvision import transforms
import torch.utils.data as data


def default_loader(path):
    return Image.open(path).convert('RGB')


def single_load(path, args):
    img = default_loader(path)
    if args.resize != 0:
        img = img.resize([args.resize, args.resize])
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    if args.type == 64:
        img_tensor = img_tensor.double()
    return img_tensor


def load_img_tensor(img_path):
    img = default_loader(img_path)
    # img = img.resize([512, 512])
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    return img_tensor


def get_img_path_list(txt_path):
    img_path_list = []
    f = open(txt_path)
    line = f.readline()
    while line:
        line = line.strip('\n')
        img_path_list.append(line)
        line = f.readline()
    f.close()
    return img_path_list


# class TestDataset(data.Dataset):
#     def __init__(self, c_dir, s_dir, config):
#         super(TestDataset, self).__init__()
#         self.c_dir = c_dir
#         self.s_dir = s_dir
#         self.config = config
#         self.c_list = [x for x in os.listdir(self.c_dir)]
#         self.s_list = [x for x in os.listdir(self.s_dir)]
#         min_len = min(len(self.c_list), len(self.s_list))
#         self.c_list = self.c_list[0:min_len]
#         self.s_list = self.s_list[0:min_len]
#
#     def __getitem__(self, index):
#         c_path = os.path.join(self.c_dir, self.c_list[index])
#         s_path = os.path.join(self.s_dir, self.s_list[index])
#
#         img = default_loader(c_path)
#         img = img.resize(self.config['img_size'])
#         c_tensor = transforms.ToTensor()(img)
#         if self.config['type'] == 64:
#             c_tensor = c_tensor.double()
#         c_name = self.c_list[index].split('.')[0]
#
#         img = default_loader(s_path)
#         img = img.resize(self.config['img_size'])
#         s_tensor = transforms.ToTensor()(img)
#         if self.config['type'] == 64:
#             s_tensor = s_tensor.double()
#         s_name = self.s_list[index].split('.')[0]
#
#         return c_tensor, s_tensor, c_name, s_name
#
#     def __len__(self):
#         return len(self.c_list)


class Dataset(data.Dataset):
    def __init__(self, args, text_path):
        super(Dataset, self).__init__()
        self.args = args
        self.text_path = text_path
        self.transform = transforms.Compose([
            transforms.Resize(self.args.fineSize),
            transforms.RandomResizedCrop(self.args.fineSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.img_list = get_img_path_list(self.text_path)

    def __getitem__(self, index):
        img_path = os.path.join(self.args.data_dir, self.img_list[index])
        img = default_loader(img_path)
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        return len(self.img_list)


class TrainDataset(data.Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args
        self.transform = transforms.Compose([
            transforms.Resize(self.args.load_size),
            transforms.RandomResizedCrop(self.args.fineSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.img_list = get_img_path_list(self.args.train_txt_path)

    def __getitem__(self, index):
        img_path = os.path.join(self.args.data_dir, self.img_list[index])
        img = default_loader(img_path)
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        return len(self.img_list)


class TestDataset(data.Dataset):
    def __init__(self, args):
        super(TestDataset, self).__init__()
        self.args = args
        self.transform = transforms.Compose([
            transforms.Resize(self.args.load_size),
            transforms.RandomResizedCrop(self.args.fineSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.img_list = get_img_path_list(self.args.test_txt_path)

    def __getitem__(self, index):
        img_path = os.path.join(self.args.data_dir, self.img_list[index])
        img = default_loader(img_path)
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        return len(self.img_list)
