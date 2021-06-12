import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import torch
import torch.utils.data as torch_data
import torch.optim as torch_optim
import torch.nn as torch_nn
import random

"""
0 - disgusted
1 - angry
2 - surprised
3 - sad
4 - neutral
5 - happy
6 - fearful
"""
"""
data=[]
data_len=10000

path=os.path.abspath('.')+'/Abstract/Male_front/disgusted.npz'
print(len(np.load(path,encoding='bytes',allow_pickle=True)))
data.append(np.load(path,encoding='bytes',allow_pickle=True)[0:data_len])
random.shuffle(data[len(data)-1])

path=os.path.abspath('.')+'/Abstract/Male_front/angry.npz'
print(len(np.load(path,encoding='bytes',allow_pickle=True)))
data.append(np.load(path,encoding='bytes',allow_pickle=True)[0:data_len])
random.shuffle(data[len(data)-1])

path=os.path.abspath('.')+'/Abstract/Male_front/surprised.npz'
print(len(np.load(path,encoding='bytes',allow_pickle=True)))
data.append(np.load(path,encoding='bytes',allow_pickle=True)[0:data_len])
random.shuffle(data[len(data)-1])

path=os.path.abspath('.')+'/Abstract/Male_front/sad.npz'
print(len(np.load(path,encoding='bytes',allow_pickle=True)))
data.append(np.load(path,encoding='bytes',allow_pickle=True)[0:data_len])
random.shuffle(data[len(data)-1])

path=os.path.abspath('.')+'/Abstract/Male_front/neutral.npz'
print(len(np.load(path,encoding='bytes',allow_pickle=True)))
data.append(np.load(path,encoding='bytes',allow_pickle=True)[0:data_len])
random.shuffle(data[len(data)-1])

path=os.path.abspath('.')+'/Abstract/Male_front/happy.npz'
print(len(np.load(path,encoding='bytes',allow_pickle=True)))
data.append(np.load(path,encoding='bytes',allow_pickle=True)[0:data_len])
random.shuffle(data[len(data)-1])

path=os.path.abspath('.')+'/Abstract/Male_front/fearful.npz'
print(len(np.load(path,encoding='bytes',allow_pickle=True)))
data.append(np.load(path,encoding='bytes',allow_pickle=True)[0:data_len])
random.shuffle(data[len(data)-1])
"""
"""
def fig2data(fig):
    import PIL.Image as Image
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w,h=fig.canvas.get_width_height()
    buf=np.frombuffer(fig.canvas.tostring_argb(),dtype=np.uint8)
    buf.shape=(w,h,4)
 
    buf=np.roll(buf,3,axis=2)
    image=Image.frombytes("RGBA",(w, h),buf.tostring())
    image=np.asarray(image)
    return image

def draw_sketch(label,u):
    ux=0
    uy=0
    sta=1
    fig=plt.figure(figsize=(0.68,0.68))
    fig_plt=fig.add_subplot(111)
    for pos in data[label][u]:
        vx=ux-pos[0]
        vy=uy-pos[1]
        if sta==0:
            fig_plt.plot([ux,vx],[uy,vy],\
                     linewidth=1,\
                     color='#000000')
        
        ux=vx
        uy=vy
        sta=pos[2]
    
    plt.xticks([])
    plt.yticks([])
    ax=plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    
    #plt.show()
    
    IMG=fig2data(fig)
    plt.close()
    img=np.zeros((48,48))
    for i in range(0,48):
        for j in range(0,48):
            if IMG[i][j][0]<128:
                img[i][j]=0
            else:
                img[i][j]=255
    
    return img
"""
# 验证模型在数据集上的正确率
def validate(model,dataset,batch_size):
    val_loader=torch_data.DataLoader(
                   dataset=dataset,
                   batch_size=batch_size,
                   #num_workers=2
               )
    result,num=0.0,0
    for images,labels in val_loader:
        pred=model.forward(images)
        pred=np.argmax(pred.data.numpy(),axis=1)
        labels=labels.data.numpy()
        result+=np.sum((pred==labels))
        num+=len(images)

    acc=result/num
    return acc

class FaceDataset(torch_data.Dataset):
    def __init__(self,st,ed):
        super(FaceDataset,self).__init__()
        self.st=st
        self.ed=ed
        self.data=[]
        for label in range(0,7):
            for u in range(st,ed+1):
                path=os.path.abspath('.')+'/dataset/'+str(label)+'/{0}.jpg'.format(u)
                self.data.append([path,label])
            
        random.shuffle(self.data)

    # 读取某幅图片, item为索引号
    def __getitem__(self,item):
        path,label=self.data[item]
        img=cv2.imread(path,0)
        img=img.reshape(1,48,48)/255.0
        img_tensor=torch.from_numpy(img)
        img_tensor=img_tensor.type('torch.FloatTensor')

        return img_tensor,label

    # 获取数据集样本个数
    def __len__(self):
        return (self.ed-self.st+1)*6

class FaceCNN(torch_nn.Module):
    def __init__(self):
        super(FaceCNN,self).__init__()

        # 第一次卷积， 池化
        self.conv1=torch_nn.Sequential(
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，
            # 卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 1, 48, 48),
            # output:(bitch_size, 64, 48, 48), (48-3+2*1)/1+1 = 48
            # 卷积层
            torch_nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            
            # 数据归一化处理，使得数据在Relu之前不会因为数据过大而导致网络性能不稳定
            # 做归一化让数据形成一定区间内的正态分布
            # 不做归一化会导致不同方向进入梯度下降速度差距很大
            torch_nn.BatchNorm2d(num_features=64),      # 归一化可以避免出现梯度散漫的现象，便于激活。
            torch_nn.RReLU(inplace=True),     # 激活函数


            torch_nn.MaxPool2d(kernel_size=2,stride=2), # 最大值池化# output(bitch_size, 64, 24, 24)
        )

        # 第二次卷积， 池化
        self.conv2=torch_nn.Sequential(
            # input:(bitch_size, 64, 24, 24), output:(bitch_size, 128, 12, 12),
            torch_nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            torch_nn.BatchNorm2d(num_features=128),
            torch_nn.RReLU(inplace=True),

            torch_nn.MaxPool2d(kernel_size=2,stride=2),
        )

        # 第三次卷积， 池化
        self.conv3=torch_nn.Sequential(
            # input:(bitch_size, 128, 12, 12), output:(bitch_size, 256, 12, 12),
            torch_nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            torch_nn.BatchNorm2d(num_features=256),
            torch_nn.RReLU(inplace=True),

            torch_nn.MaxPool2d(kernel_size=2,stride=2),
            # 最后一层不需要添加激活函数
        )

        # 全连接层
        self.fc=torch_nn.Sequential(
            torch_nn.Dropout(p=0.2),
            torch_nn.Linear(in_features=256*6*6,out_features=4096),
            torch_nn.RReLU(inplace=True),

            torch_nn.Dropout(p=0.5),
            torch_nn.Linear(in_features=4096,out_features=1024),
            torch_nn.RReLU(inplace=True),

            torch_nn.Linear(in_features=1024,out_features=256),
            torch_nn.RReLU(inplace=True),

            torch_nn.Linear(in_features=256,out_features=7),
        )

    # 前向传播
    # 使用sequential模块后无需再在forward函数中添加激活函数
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)

        # 数据扁平化
        x=x.view(x.shape[0], -1)   # 输出维度，-1表示该维度自行判断
        y=self.fc(x)
        return y
    
def predict(model,dataset,u):
    img,label=dataset[u]
    plt.imshow(img[0].numpy(),cmap='gray')
    img=img.unsqueeze(dim=0)
    pred=model(img)
    pred=np.argmax(pred.data.numpy(),axis=1)
    print("label:",label,
          "\npred:",pred[0])
        
    
# 训练模型
def train(train_dataset,test_dataset,batch_size,epochs,LR):
    # 载入数据并分割batch
    train_loader=torch_data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        #num_workers=2
    )
    # 构建模型
    model=FaceCNN()
    # 损失函数
    loss_function=torch_nn.CrossEntropyLoss()
    # 优化器
    optimizer=torch_optim.Adam(model.parameters(),lr=LR)

    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate=0
        # scheduler.step()
        # 注意dropout网络结构下训练和test模式下是不一样的结构
        model.train()  # 模型训练，调用Modlue类提供的train()方法切换到train状态
        for images,labels in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output=model.forward(images)
            # 误差计算
            loss_rate=loss_function(output,labels)
            # 误差的反向传播
            loss_rate.backward()
            # 更新参数
            optimizer.step()

        # 打印每轮的损失
        print('After {} epochs , '.format(epoch+1))
        print('After {} epochs , the loss_rate is : '.format(epoch+1),loss_rate.item())
        if epoch%3==0:
            model.eval()  # 模型评估,切换到test状态继续执行
            acc_train=validate(model,train_dataset,batch_size)
            acc_test=validate(model,test_dataset,batch_size)
            print('After {} epochs , the acc_train is : '.format(epoch+1),acc_train)
            print('After {} epochs , the acc_test is : '.format(epoch+1),acc_test)

    return model    

def pre_process(st,ed):
    path=os.path.abspath('.')+'/dataset/'
    for label in range(6,7):
        print(label)
        for i in range(st,ed+1):
            cv2.imwrite(path+str(label)+'/{0}.jpg'.format(i),draw_sketch(label,i))
            if i%500==0:
                print(i)
        
        print(label,' FIN!\n')
"""
pre_process将npz数据绘制成48*48的jpg格式图片储存在dataset文件夹中
只需要运行一次
"""
IFPRE=0
IFTRAIN=0
if IFPRE:
    pre_process(0,4859);
    print("PRE_PROCESS FIN!");

train_dataset=FaceDataset(0,10)
test_dataset=FaceDataset(11,15)

if IFTRAIN:
    model=train(train_dataset,test_dataset,batch_size=128,epochs=7,LR=0.1)
    torch.save(model,'model_net1.pkl')
    
model=torch.load('model_net1.pkl')
predict(model,test_dataset,1)

"""
0 - disgusted
1 - angry
2 - surprised
3 - sad
4 - neutral
5 - happy
6 - fearful
"""