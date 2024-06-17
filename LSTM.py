import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import warnings#避免一些可以忽略的报错
warnings.filterwarnings('ignore')#filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别.
import random#设置随机种子
from sklearn.preprocessing import MinMaxScaler


class Config():
    data_path = 'DataSet/DailyDelhiClimateTrain.csv'
    timestep = 12  # 时间步长，就是利用多少时间窗口
    batch_size = 64  # 批次大小
    feature_size = 1  # 每个步长对应的特征数量，这里只使用1维
    hidden_size = 64  # 隐层大小
    out_channels = 50  # CNN输出通道
    num_heads = 1  # 注意力机制头的数量
    output_size = 1  # 由于是单输出任务，最终输出层大小为1
    num_layers = 5  # lstm的层数
    epochs = 500  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.0001  # 学习率
    model_name = 'LSTM model'  # 模型名称
    save_path = '{}.pth'.format(model_name)  # 最优模型保存路径

def split_data(data,time_step=12):
    dataX=[]
    datay=[]
    for i in range(len(data)-time_step):
        dataX.append(data[i:i+time_step])
        datay.append(data[i+time_step])
    dataX=np.array(dataX).reshape(len(dataX),time_step,-1)
    datay=np.array(datay)
    return dataX,datay

# 划分训练集和测试集的函数
def train_test_split(dataX, datay, shuffle=True, percentage=0.8):
    """
    将训练数据X和标签y以numpy.array数组的形式传入
    划分的比例定为训练集:测试集=8:2
    """
    if shuffle:
        random_num = [index for index in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX = dataX[random_num]
        datay = datay[random_num]
    split_num = int(len(dataX) * percentage)
    train_X = dataX[:split_num]
    train_y = datay[:split_num]
    test_X = dataX[split_num:]
    test_y = datay[split_num:]
    return train_X, train_y, test_X, test_y

def mse(pred_y,true_y):
    return np.mean((pred_y-true_y) ** 2)

# 定义LSTM模型类
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # 初始化记忆状态c0
        # print(f"x.shape:{x.shape},h0.shape:{h0.shape},c0.shape:{c0.shape}")
        out, _ = self.lstm(x, (h0, c0))  # LSTM前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出作为预测结果
        return out


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True#将cudnn框架中的随机数生成器设为确定性模式
    torch.backends.cudnn.benchmark = False#关闭CuDNN框架的自动寻找最优卷积算法的功能，以避免不同的算法对结果产生影响
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    config = Config()

    train_df=pd.read_csv(config.data_path)
    meantemp = train_df['meantemp'].values
    plt.plot([i for i in range(len(meantemp))], meantemp)
    plt.show()
    scaler = MinMaxScaler()
    # 将数据进行归一化
    meantemp = scaler.fit_transform(meantemp.reshape(-1, 1))

    dataX, datay = split_data(meantemp, time_step=config.timestep)
    train_X, train_y, test_X, test_y = train_test_split(dataX, datay, shuffle=False, percentage=0.8)
    X_train, y_train = train_X, train_y

    test_X1 = torch.Tensor(test_X)
    test_y1 = torch.Tensor(test_y)

    # 定义输入、隐藏状态和输出维度
    input_size = config.feature_size  # 输入特征维度
    hidden_size = config.hidden_size  # LSTM隐藏状态维度
    num_layers = config.num_layers  # LSTM层数
    output_size = config.output_size  # 输出维度（预测目标维度）

    # 创建LSTM模型实例
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # 训练周期为500次
    num_epochs = config.epochs
    batch_size = config.batch_size  # 一次训练的数量
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    # 损失函数
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    print(f"start")
    for epoch in range(num_epochs):

        random_num = [i for i in range(len(train_X))]
        np.random.shuffle(random_num)

        train_X = train_X[random_num]
        train_y = train_y[random_num]

        train_X1 = torch.Tensor(train_X[:batch_size])
        train_y1 = torch.Tensor(train_y[:batch_size])

        # 训练
        model.train()
        # 将梯度清空
        optimizer.zero_grad()
        # 将数据放进去训练
        output = model(train_X1)
        # 计算每次的损失函数
        train_loss = criterion(output, train_y1)
        # 反向传播
        train_loss.backward()
        # 优化器进行优化(梯度下降,降低误差)
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                output = model(test_X1)
                test_loss = criterion(output, test_y1)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"epoch:{epoch},train_loss:{train_loss},test_loss:{test_loss}")
    print('Finished Training')

    train_X1 = torch.Tensor(X_train)
    train_pred = model(train_X1).detach().numpy()
    test_pred = model(test_X1).detach().numpy()
    pred_y = np.concatenate((train_pred, test_pred))
    pred_y = scaler.inverse_transform(pred_y).T[0]
    true_y = np.concatenate((y_train, test_y))
    true_y = scaler.inverse_transform(true_y).T[0]
    print(f"mse(pred_y,true_y):{mse(pred_y, true_y)}")

    plt.title(config.model_name)
    x = [i for i in range(len(true_y))]
    plt.plot(x, pred_y, marker="o", markersize=1, label="pred_y")
    plt.plot(x, true_y, marker="x", markersize=1, label="true_y")
    plt.legend()
    plt.show()