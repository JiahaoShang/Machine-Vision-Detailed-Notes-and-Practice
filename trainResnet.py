from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from time import strftime, gmtime

import random,shutil

from torchvision import models

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


def getData(dType="train",dataDir="",bSize=8,shuffle=True,imgH=224,imgW=224,channels=3):
    # norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #
    # transform = T.Compose([T.Resize((imgH, imgW)), T.ToTensor(),norm])  # img = img /2+0.5
    if channels==1:
        transform = T.Compose([T.Resize((imgH, imgW)),T.Grayscale(num_output_channels=1),
                               T.ToTensor()])  # img = 0-1
    else:
        transform = T.Compose([T.Resize((imgH, imgW)),T.ToTensor()])  # img = 0-1

    if dType=="train":
        trainDSet = ImageFolder(dataDir + "/train/", transform=transform)
        dLoad = DataLoader(trainDSet,
                           batch_size=bSize,
                           shuffle=shuffle,
                           num_workers=4)
        return dLoad
    if dType=="test":
        valDSet = ImageFolder(dataDir + "/test/", transform=transform)
        dLoad = DataLoader(valDSet,
                           batch_size=bSize,
                           shuffle=False,
                           num_workers=4)
        return dLoad



def buildNet(class_num=11,name="resnet18",input_channels=3,ptrain=False):
    if name=="resnet18":
        model = models.resnet18(pretrained=ptrain)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False)
        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in, class_num)
    if name=="resnet34":
        model = models.resnet34(pretrained=ptrain)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False)
        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in, class_num)
    if name=="resnet50":
        model = models.resnet50(pretrained=ptrain)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False)
        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in, class_num)
    return model



def train(opt):
    # saveDir = genDIr()
    saveDir = "./trainLog/"

    model = buildNet(class_num=opt.clsNum,name=opt.saveName,input_channels=opt.imageC,ptrain=opt.ptrain)

    trainData = getData("train", dataDir=opt.dataDir, bSize=opt.trainBSize, imgH=opt.imageH, imgW=opt.imageW,channels=opt.imageC)
    testData = getData("test", dataDir=opt.dataDir, bSize=opt.testBSize, imgH=opt.imageH, imgW=opt.imageW,channels=opt.imageC)
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device=",device)

    model = model.to(device)
    loss_fc = nn.CrossEntropyLoss()
    if opt.optimizer=="sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    if opt.optimizer=="adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr)
    trainLoss = []
    trainAcc = []
    valLoss = []
    testAcc = []
    best_acc = 0
    time_start = time.time()
    for epoch in range(opt.epochs):
        trainLoss.append(0)
        trainBN = 0
        model.train()
        train_correct = 0
        train_total = 0
        for sample_batch in tqdm(trainData,ncols=50):
            inputs = sample_batch[0]
            train_labels = sample_batch[1]
            model.train()
            # GPU/CPU
            inputs = inputs.to(device)
            train_labels = train_labels.to(device)
            inputs, labels = Variable(inputs), Variable(train_labels)
            #梯度清零
            optimizer.zero_grad()

            train_outputs = model(inputs)
            loss = loss_fc(train_outputs, train_labels)
            _, train_prediction = torch.max(train_outputs, 1)
            train_correct += (torch.sum((train_prediction == train_labels))).item()
            train_total += train_labels.size(0)

            trainLoss[-1] = trainLoss[-1]+loss.item()
            trainBN += 1

            # loss求导，反向
            loss.backward()
            # 优化
            optimizer.step()

        train_acc = train_correct / train_total
        trainAcc.append(train_acc)
            # if i%100==0:
        # print()
        print('[{}/{}] [TrainLoss {:.5f}] [TrainAcc {:.5f}]'.format(epoch+1,opt.epochs,loss.item(),train_acc))

        trainLoss[-1] /= trainBN
        # lr_scheduler.step()

        # 測試
        # if epoch % 1 == 0:
        model.eval()
        correct = 0
        total = 0
        epo_count = 0
        test_loss0 = []
        valLoss.append(0)
        with torch.no_grad(): #不需要梯度计算
            for images_test, labels_test in testData:
                epo_count = epo_count + 1
                images_test = images_test.to(device)
                labels_test = labels_test.to(device)
                outputs_test = model(images_test)
                test_loss = loss_fc(outputs_test, labels_test)
                test_loss0.append(test_loss.item())
                a = valLoss.pop()
                a += test_loss.item()
                valLoss.append(a)
                _, prediction = torch.max(outputs_test, 1)
                correct += (torch.sum((prediction == labels_test))).item()
                total += labels_test.size(0)
            b = valLoss.pop()
            b /= epo_count
            valLoss.append(b)
            test_acc =  correct / total
            testAcc.append(test_acc)
            torch.cuda.empty_cache()  #释放显存
            print('[{}] [TestLoss {:.5f}] [TestAcc {:.5f}]'.format(
                epoch + 1, np.mean(test_loss0), correct / total))
        if test_acc>best_acc:
            best_acc = test_acc
            # torch.save(model,"./trainLog/best.pth")

    time_end = time.time()
    print("训练时间：{:.2f} 秒".format(time_end-time_start))

    acc_loss = np.array([correct / total, np.mean(test_loss0)])
    np.save("./resnet_accLoss.npy", acc_loss)
    print("准确率：{:.2f}%".format((correct / total) * 100))
    print("损失值：{:.4f}".format(np.mean(test_loss0)))

    # print('training finish !')
    torch.save(model, "./trainLog/resnet.pth") #保存结构和参数

    plt.plot(testAcc,label="testAcc")
    plt.plot(trainAcc, label="trainAcc")
    plt.legend()
    plt.savefig(saveDir+"resnet_acc.png")
    plt.close()

    plt.clf()
    plt.plot(valLoss, label="testLoss")
    plt.plot(trainLoss, label="trainLoss")
    plt.legend()
    plt.savefig(saveDir + "resnet_loss.png")
    plt.close()


def dataSpilt(opt=None):
    trainDir = opt.dataDir +"train/"
    testDir = opt.dataDir + "test/"
    if not os.path.exists(testDir):
        os.mkdir(testDir)
        print("mkdir",testDir)
    clsDirs = os.listdir(trainDir)
    for clsDir in clsDirs:
        if not os.path.exists(testDir+clsDir):
            os.mkdir(testDir+clsDir)

    # test to train
    testCLSDirs = os.listdir(testDir)
    for d in testCLSDirs:
        files = os.listdir(testDir + d)
        for file in files:
            src = testDir + d + "/" + file
            dst = trainDir + d + "/" + file
            shutil.move(src, dst)

    # train to test
    for d in clsDirs:
        files = os.listdir(trainDir + d)
        test_nums = int(len(files)*opt.rate[1])
        random.shuffle(files)
        for i,file in enumerate(files):
            if i>test_nums:
                continue
            else:
                src = trainDir + d + "/" + file
                dst = testDir + d + "/" + file
                shutil.move(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    # resnet18, resnet34, resnet50
    parser.add_argument('--saveName', type=str, default="resnet50", help="模型名称")
    parser.add_argument('--dataDir', type=str, default="./dataset/")
    parser.add_argument('--testDir', type=str, default="./dataset/")
    parser.add_argument('--trainBSize', type=int, default=64)
    parser.add_argument('--testBSize', type=int, default=16)

    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--clsNum', type=int, default=2)
    parser.add_argument('--imageH', type=int, default=224) #
    parser.add_argument('--imageW', type=int, default=224)
    parser.add_argument('--imageC', type=int, default=3)
    parser.add_argument('--optimizer', type=str, default="adam",help="sgd or adam")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.95)#momentum
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--ptrain', type=bool, default=True)
    parser.add_argument('--rate', type=list, default=[0.8,0.2])

    opt = parser.parse_args()

    # dataSpilt(opt)

    # 建立文件夹
    if not os.path.exists("./trainLog"):
        os.mkdir("./trainLog")

    train(opt)

