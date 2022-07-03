from dataset import DMD
from tqdm import tqdm
import numpy as np
from model import resnet18 as Net1
from model import resnet9 as Net2
from model import resnext1 as Net3
import argparse 
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

def train(model, train_loader, optimizer, epoch):
    # model.train() 
    total_acc = 0
    total_num = 0
    losses = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for images, labels in pbar:
        # images = images.float()
        output = model(images)
        # output = output.long()
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, labels)
        # print(output)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 
        pred = output.argmax(dim=1)
        # print(pred)
        # print(labels)
        acc = torch.eq(pred,labels).sum().float().item()
        # print(acc)
        total_acc += acc
        total_num += labels.shape[0] 
        losses += loss
        pbar.set_description(f'Epoch {epoch} [TRAIN] loss = {loss:.2f}, acc = {100 * (acc / labels.shape[0]):.2f} %')

best_acc = -1.0
def evaluate(model, val_loader, epoch=0, save_path='./best_model.pkl'):
    # model.eval()
    global best_acc
    total_acc = 0
    total_num = 0

    for images, labels in val_loader:
        # images = images.float()
        output = model(images)
        # output = output.long()
        pred = output.argmax(dim=1)
        acc = torch.eq(pred,labels).sum().float().item()
        total_acc += acc
        total_num += labels.shape[0] 
    acc = total_acc / total_num 
    if acc > best_acc:
        best_acc = acc
        # model.save(save_path)
        torch.save(model, 'best_model.pt')
    print ('Test in epoch', epoch, 'Accuracy is', acc, 'Best accuracy is', best_acc)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--num_classes', type=int, default=4)

    # parser.add_argument('--lr', type=float, default=2e-3)
    # parser.add_argument('--weight_decay', type=float, default=1e-5)

    # parser.add_argument('--resume', type=bool, default=False)
    # parser.add_argument('--eval', type=bool, default=False)

    # parser.add_argument('--dataroot', type=str, default='/home/merlin/DM/Project')
    # parser.add_argument('--model_path', type=str, default='./best_model.pkl')

    # args = parser.parse_args()

    root_dir = '/home/merlin/DM/Project'
    lr = 2e-3
    epochs = 100
    weight_decay = 1e-3
    num_classes = 4
    train_dataset = DMD(root_dir, part='train')
    val_dataset = DMD(root_dir, part='val')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    model1 = Net1(num_classes=num_classes)
    model2 = Net2(num_classes=num_classes)
    model3 = Net3(num_classes=num_classes)

    # model1 = torch.load("./best_model_resnet18.pt")
    model2 = torch.load("./best_model_resnet9.pt")
    # model3 = torch.load("./best_model_resnext1.pt")
    
    optimizer = optim.SGD(model2.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) 
    # optimizer = optim.Rprop(model2.parameters(), lr=lr) 
    # if args.resume:
    #     model.load(args.model_path)
    # if args.eval:
    #     evaluate(model, val_loader)
    #     return
    for epoch in range(epochs):
        train(model2, train_loader, optimizer, epoch)
        evaluate(model2, val_loader, epoch)

    # evaluate(model1, val_loader)
    # evaluate(model2, val_loader)
    # evaluate(model3, val_loader)

if __name__ == '__main__':
    main()
