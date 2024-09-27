import os
import torch
from torch import nn
from torch.nn import Conv2d, Linear, ReLU
from torch.nn import MaxPool2d
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


# Dataset:创建数据集的函数；__init__:初始化数据内容和标签
# __geyitem:获取数据内容和标签
# __len__:获取数据集大小
# daataloader:数据加载类，接受来自dataset已经加载好的数据集
# torchbision:图形库，包含预训练模型，加载数据的函数、图片变换，裁剪、旋转等
# torchtext:处理文本的工具包，将不同类型的额文件转换为datasets

# 预处理：将两个步骤整合在一起
transform = transforms.Compose({
    transforms.ToTensor(),  # 将灰度图片像素值（0~255）转为Tensor（0~1），方便后续处理
    # transforms.Normalize((0.1307,),(0.3081)),    # 归一化，均值0，方差1;mean:各通道的均值std：各通道的标准差inplace：是否原地操作
})

# 加载数据集
# 训练数据集
train_data = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# transform：指示加载的数据集应用的数据预处理的规则，shuffle：洗牌，是否打乱输入数据顺序
# 测试数据集
test_data = MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度：{}".format(train_data_size))
print("测试数据集的长度：{}".format(test_data_size))

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = MaxPool2d(2)
        self.linear1 = Linear(320, 128)
        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 10)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x

# 损失函数CrossentropyLoss
model = MnistModel()#实例化
criterion = nn.CrossEntropyLoss()   # 交叉熵损失，相当于Softmax+Log+NllLoss
# 线性多分类模型Softmax,给出最终预测值对于10个类别出现的概率，Log:将乘法转换为加法，减少计算量，保证函数的单调性
# NLLLoss:计算损失，此过程不需要手动one-hot编码，NLLLoss会自动完成

# SGD，优化器，梯度下降算法e
optimizer = torch.optim.SGD(model.parameters(), lr=0.14)#lr:学习率

# 保存模型到model文件夹
os.makedirs('model', exist_ok=True)  # 确保文件夹存在
torch.save(model.state_dict(), 'model/model.pkl')

# 加载模型
loaded_model = MnistModel()
loaded_model.load_state_dict(torch.load('model/model.pkl'))
loaded_model.eval()  # 设置模型为评估模式
