from dataset import DMD_sub
from tqdm import tqdm
from model import resnet18 as Net1
from model import resnet9 as Net2
from model import resnext1 as Net3
import argparse 
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

def submit(model1, model2, model3, sub_loader, batch_size, num_classes):
    otp=[]
    id_stt = 100000

    for images in sub_loader:
        # images = images.float()
        output1 = model1(images)
        output2 = model2(images)
        output3 = model3(images)
        
        output1 = output1.long()
        pred1 = output1.argmax(dim=1)
        output2 = output2.long()
        pred2 = output2.argmax(dim=1)
        output3 = output3.long()
        pred3 = output3.argmax(dim=1)

        std = 1 / 3
        for i in range(pred1.shape[0]):
            otp_temp = [id_stt,0,0,0,0]
            if pred1[i] == 0:
                # otp.append([id_stt,0.5,0,0,0])
                otp_temp[1] += std
            elif pred1[i] == 1:
                # otp.append([id_stt,0,0.5,0,0])
                otp_temp[2] += std
            elif pred1[i] == 2:
                # otp.append([id_stt,0,0,0.5,0])
                otp_temp[3] += std
            elif pred1[i] == 3:
                # otp.append([id_stt,0,0,0,0.5])
                otp_temp[4] += std
                
            if pred2[i] == 0:
                # otp.append([id_stt,0.5,0,0,0])
                otp_temp[1] += std
            elif pred2[i] == 1:
                # otp.append([id_stt,0,0.5,0,0])
                otp_temp[2] += std
            elif pred2[i] == 2:
                # otp.append([id_stt,0,0,0.5,0])
                otp_temp[3] += std
            elif pred2[i] == 3:
                # otp.append([id_stt,0,0,0,0.5])
                otp_temp[4] += std

            if pred3[i] == 0:
                # otp.append([id_stt,0.5,0,0,0])
                otp_temp[1] += std
            elif pred3[i] == 1:
                # otp.append([id_stt,0,0.5,0,0])
                otp_temp[2] += std
            elif pred3[i] == 2:
                # otp.append([id_stt,0,0,0.5,0])
                otp_temp[3] += std
            elif pred3[i] == 3:
                # otp.append([id_stt,0,0,0,0.5])
                otp_temp[4] += std
            
            pred = otp_temp.index(max(otp_temp[1:]))

            if pred == 1:
                otp.append([id_stt,1,0,0,0])
            elif pred == 2:
                otp.append([id_stt,0,1,0,0])
            elif pred == 3:
                otp.append([id_stt,0,0,1,0])
            elif pred == 4:
                otp.append([id_stt,0,0,0,1])
            
            id_stt = id_stt + 1
            # print(id_stt)

    submit = pd.DataFrame(otp, columns=['id','label_0','label_1','label_2','label_3'])

    submit.to_csv(("./push.csv"), index=False) 

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
    num_classes = 4
    sub_dataset = DMD_sub(root_dir)
    sub_loader = DataLoader(sub_dataset, batch_size=64, shuffle=False)

    model1 = Net1(num_classes=num_classes)
    model2 = Net2(num_classes=num_classes)
    model3 = Net3(num_classes=num_classes)

    # model1 = torch.load("./best_model.pt")
    model1 = torch.load("./best_model_resnet18.pt")
    model2 = torch.load("./best_model_resnet9.pt")
    model3 = torch.load("./best_model_resnext1.pt")

    # optimizer = optim.SGD(model3.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) 
    # if args.resume:
    #     model.load(args.model_path)
    # if args.eval:
    #     evaluate(model, val_loader)
    #     return
    
    submit(model1, model2, model3, sub_loader, batch_size=64, num_classes=4)

if __name__ == '__main__':
    main()
