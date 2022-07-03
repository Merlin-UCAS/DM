import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd


class DMD(Dataset):
    def __init__(self, root_dir, part='train'):
        super().__init__()
        self.root_dir = root_dir
        self.num_classes = 4
        self.part = part
        self.image_list = []
        self.id_list = []
        # if part == 'train':
        #     list_path = os.path.join(self.root_dir, 'datasets/train.csv')
        # elif part == 'val':
        #     list_path = os.path.join(self.root_dir, 'datasets/testA.csv')

        list_path = os.path.join(self.root_dir, 'datasets/train.csv')
        train = pd.read_csv(list_path)

        train_list = []
        for items in train.values:
            train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [int(items[2])])
        train = pd.DataFrame(np.array(train_list))
        train.columns = ['id'] + ['s_' + str(i) for i in range(len(train_list[0])-2)] + ['label']

        y_train = train['label']
        x_train = train.drop(['id', 'label'], axis=1)
        print(x_train.shape, y_train.shape)

        # train.head()
        # train.head()
        # train.describe()
        # dmd.describe()
        # train.info()
        # dmd.info()

        x_train_0=[]
        y_train_0=[]
        x_train_1=[]
        y_train_1=[]
        x_train_2=[]
        y_train_2=[]
        x_train_3=[]
        y_train_3=[]

        for i in range(x_train.shape[0]):
            if y_train[i] == 0:
                x_train_0.append(x_train.loc[i])
                y_train_0.append(y_train.loc[i])
            elif y_train[i] == 1:
                x_train_1.append(x_train.loc[i])
                y_train_1.append(y_train.loc[i])
            elif y_train[i] == 2:
                x_train_2.append(x_train.loc[i])
                y_train_2.append(y_train.loc[i])
            elif y_train[i] == 3:
                x_train_3.append(x_train.loc[i])
                y_train_3.append(y_train.loc[i])

        x_train_0_np = np.array(x_train_0)
        x_train_1_np = np.array(x_train_1)
        x_train_2_np = np.array(x_train_2)
        x_train_3_np = np.array(x_train_3)

        print(x_train_0_np.shape)
        print(x_train_1_np.shape)
        print(x_train_2_np.shape)
        print(x_train_3_np.shape)

        x_train_pp=[]
        y_train_pp=[]
        tt_num = max(x_train_0_np.shape[0], x_train_1_np.shape[0], x_train_2_np.shape[0], x_train_3_np.shape[0])

        for i in range(tt_num):
            x_train_pp.append(x_train_0[i])
            y_train_pp.append(y_train_0[i])

            x_train_pp.append(x_train_1[random.randrange(0, x_train_1_np.shape[0])])
            y_train_pp.append(y_train_1[0])

            x_train_pp.append(x_train_2[random.randrange(0, x_train_2_np.shape[0])])
            y_train_pp.append(y_train_2[0])
            
            x_train_pp.append(x_train_3[random.randrange(0, x_train_3_np.shape[0])])
            y_train_pp.append(y_train_3[0])

        # random.seed(2022)
        # random.shuffle(x_train_pp)
        # random.seed(2022)
        # random.shuffle(y_train_pp)

        x_train_pp_np = np.array(x_train_pp)
        y_train_pp_np = np.array(y_train_pp)

        if part == 'val':
            self.image_list=x_train_pp_np.tolist()[0:int(tt_num/10)]
            self.id_list=y_train_pp_np.tolist()[0:int(tt_num/10)]
        elif part == 'train':
            self.image_list=x_train_pp_np.tolist()[int(tt_num/10):]
            self.id_list=y_train_pp_np.tolist()[int(tt_num/10):]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.id_list[idx]
        img_temp = np.array(image)
        img_temp.resize(14,14)

        img=[]
        img.append(img_temp)
        img.append(img_temp)
        img.append(img_temp)
        img = torch.tensor(np.array(img), dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label

def test_dataset():
    root = '/home/merlin/DM/Project'
    part = 'train'

    dataset = DMD(root, part=part)
    # def __init__(self, root_dir, part='train'):
    
    for images,labels in dataset:
        # print(images, labels)
        pass 

if __name__=='__main__':
    test_dataset()


class DMD_sub(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.num_classes = 4
        self.image_list = []
        # if part == 'train':
        #     list_path = os.path.join(self.root_dir, 'datasets/train.csv')
        # elif part == 'val':
        #     list_path = os.path.join(self.root_dir, 'datasets/testA.csv')

        list_path = os.path.join(self.root_dir, 'datasets/testA.csv')
        test = pd.read_csv(list_path)

        test_list=[]
        for items in test.values:
            test_list.append([items[0]] + [float(i) for i in items[1].split(',')])
        test = pd.DataFrame(np.array(test_list))
        test.columns = ['id'] + ['s_'+str(i) for i in range(len(test_list[0])-1)]

        X_test = test.drop(['id'], axis=1)
        print(X_test.shape)

        x_test=[]

        for i in range(X_test.shape[0]):
            x_test.append(X_test.loc[i])

        x_test_np = np.array(x_test)

        print(x_test_np.shape)

        self.image_list = x_test_np.tolist()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        img_temp = np.array(image)
        img_temp.resize(14,14)

        img=[]
        img.append(img_temp)
        img.append(img_temp)
        img.append(img_temp)
        img = torch.tensor(np.array(img), dtype=torch.float)
        
        return img

def test_dataset():
    root = '/home/merlin/DM/Project'

    dataset = DMD_sub(root)
    # def __init__(self, root_dir, part='train'):
    
    for images in dataset:
        # print(images)
        pass 

if __name__=='__main__':
    test_dataset()
