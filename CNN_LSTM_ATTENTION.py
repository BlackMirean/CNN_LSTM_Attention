import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tushare as ts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
import random

class Config():
    data_path = 'DataSet/DailyDelhiClimateTrain.csv'
    timestep = 12  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    feature_size = 1  # 每个步长对应的特征数量，这里只使用1维
    hidden_size = 256  # 隐层大小
    out_channels = 50  # CNN输出通道
    num_heads = 1  # 注意力机制头的数量
    output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
    num_layers = 2  # lstm的层数
    epochs = 100  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.0001  # 学习率
    model_name = 'CNN_LSTM_ATTENTION model'  # 模型名称
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

# 定义CNN + LSTM + Attention网络
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, feature_size, timestep, hidden_size, num_layers, out_channels, num_heads, output_size):
        super(CNN_LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数
        self.conv1d = nn.Conv1d(in_channels=feature_size, out_channels=out_channels, kernel_size=3, padding=1) # 卷积层
        # LSTM层
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为 1
        self.lstm = nn.LSTM(out_channels, hidden_size, num_layers, batch_first=True)
        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_heads, batch_first=True,
                                               dropout=0.8)
        # 输出层
        self.fc1 = nn.Linear(timestep * hidden_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        x = x.transpose(1, 2)  # batch_size, feature_size, timestep[32, 1, 20]
        # 卷积运算
        output = self.conv1d(x)
        batch_size = x.shape[0]  # 获取批次大小
        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output = output.transpose(1, 2)  # batch_size, feature_size, timestep[32, 1, 20]
        # LSTM运算
        output, (h_0, c_0) = self.lstm(output, (h_0, c_0))  # batch_size, timestep, hidden_size
        # 注意力计算
        attention_output, attn_output_weights = self.attention(output, output, output)
        # 展开
        output = output.flatten(start_dim=1)
        # 全连接层
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        return output

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True  # 将cudnn框架中的随机数生成器设为确定性模式
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN框架的自动寻找最优卷积算法的功能，以避免不同的算法对结果产生影响
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    config = Config()

    # 加载时间序列数据
    train_df = pd.read_csv(config.data_path)
    # 将数据进行标准化
    meantemp = train_df['meantemp'].values
    plt.plot([i for i in range(len(meantemp))], meantemp)
    plt.show()
    scaler = MinMaxScaler()
    meantemp = scaler.fit_transform(meantemp.reshape(-1, 1))
    
    # 获取训练数据
    dataX, datay = split_data(meantemp, time_step=config.timestep)
    train_X, train_y, test_X, test_y = train_test_split(dataX, datay, shuffle=False, percentage=0.8)
    X_train, y_train = train_X, train_y

    X_train_tensor = torch.from_numpy(X_train).to(torch.float32)
    y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
    x_test_tensor = torch.from_numpy(test_X).to(torch.float32)
    y_test_tensor = torch.from_numpy(test_y).to(torch.float32)

    test_X1 = torch.Tensor(test_X)
    test_y1 = torch.Tensor(test_y)

    # 形成训练数据集
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)

    # 将数据加载成迭代器
    train_loader = torch.utils.data.DataLoader(train_data, config.batch_size, False)
    test_loader = torch.utils.data.DataLoader(test_data,  config.batch_size, False)

    model = CNN_LSTM_Attention(config.feature_size, config.timestep, config.hidden_size, config.num_layers,
                               config.out_channels, config.num_heads, config.output_size)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))  # 定义优化器

    batch_size = config.batch_size  # 一次训练的数量

    print(f"start")
    # 模型训练
    for epoch in range(config.epochs):
        random_num = [i for i in range(len(train_X))]
        np.random.shuffle(random_num)

        train_X = train_X[random_num]
        train_y = train_y[random_num]

        train_X1 = torch.Tensor(train_X[:batch_size])
        train_y1 = torch.Tensor(train_y[:batch_size])

        model.train()
        running_loss = 0
        train_bar = tqdm(train_loader)  # 形成进度条
        for data in train_bar:
            optimizer.zero_grad()
            output = model(train_X1)
            train_loss = criterion(output, train_y1)
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.6f}".format(epoch + 1, config.epochs, train_loss)

        # 模型验证
        model.eval()
        test_loss = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for data in test_bar:
                output = model(test_X1)
                test_loss = criterion(output, test_y1)

        if test_loss < config.best_loss:
            config.best_loss = test_loss
            torch.save(model.state_dict(), config.save_path)
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

