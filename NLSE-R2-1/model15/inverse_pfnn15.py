import torch
import math
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn


# 定义 Glorot Uniform 初始化函数
def initialize_weight(m):
    if isinstance(m, nn.Linear):  # 检查模块是否是 nn.Linear
        # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.xavier_normal_(m.weight, gain=1)  # Glorot Uniform 初始化
        if m.bias is not None:
            # nn.init.normal_(m.bias, std=0.2)
            nn.init.zeros_(m.bias)
        # if m.bias is not None:
        #     nn.init.xavier_uniform_(m.bias)

        # if m.bias is not None:
        #     fan_in = m.weight.size(1)  # 输入特征数量
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(m.bias, a=-bound, b=bound)


# 定义第一个子网络
class Subnet1(torch.nn.Module):
    def __init__(self, num_neurons=32):
        super(Subnet1, self).__init__()
        self.fc1 = torch.nn.Linear(2, num_neurons)
        self.fc2_1 = torch.nn.Linear(num_neurons, num_neurons // 2)
        self.fc2_2 = torch.nn.Linear(num_neurons, num_neurons // 2)
        self.fc3_1 = torch.nn.Linear(num_neurons // 2, num_neurons // 4)
        self.fc3_2 = torch.nn.Linear(num_neurons // 2, num_neurons // 4)

        self.fc3_3 = torch.nn.Linear(num_neurons // 2, num_neurons // 4)
        self.fc3_4 = torch.nn.Linear(num_neurons // 2, num_neurons // 4)

        # 输出层
        self.fc_output = torch.nn.Linear(num_neurons, 2)

        self.apply(initialize_weight)

    def forward(self, x):
        x1 = F.gelu(self.fc1(x))

        x2_1 = F.gelu(self.fc2_1(x1))
        x2_2 = F.gelu(self.fc2_2(x1))

        x3_1 = F.gelu(self.fc3_1(x2_1))
        x3_2 = F.gelu(self.fc3_2(x2_1))
        x3_3 = F.gelu(self.fc3_3(x2_2))
        x3_4 = F.gelu(self.fc3_4(x2_2))
        x3 = torch.cat((x3_1, x3_2, x3_3, x3_4), dim=1)

        x = self.fc_output(x3)

        return x


class BS_net(torch.nn.Module):
    def __init__(self, input_num=1, output_num=1, num_neurons=32, num_layers=4):
        super(BS_net, self).__init__()
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.input_num = input_num
        self.output_num = output_num

        # 定义第一层
        self.fc1 = torch.nn.Linear(input_num, num_neurons)

        # 定义隐藏层分支（从第二层开始）
        self.branches = torch.nn.ModuleDict()
        for layer in range(2, num_layers + 1):  # 层数从2开始，到num_layers结束
            num_branches = 2 ** (layer - 1)  # 当前层的分支数量是2的(layer-1)次方
            input_dim = num_neurons // (2 ** (layer - 2))  # 输入维度：前一层的输出
            output_dim = num_neurons // (2 ** (layer - 1))  # 输出维度：当前分支的输出
            # 为当前层的每个分支创建Linear层
            self.branches[f"layer_{layer}"] = torch.nn.ModuleList([
                torch.nn.Linear(input_dim, output_dim) for _ in range(num_branches)
            ])

        # 定义输出层
        self.fc_output = torch.nn.Linear(num_neurons, output_num)

        # 初始化权重
        self.apply(initialize_weight)

    def forward(self, x):
        # 第一层处理
        x1 = F.silu(self.fc1(x))

        # 初始化上一层的输出为列表
        prev_outputs = [x1]

        # 遍历隐藏层
        for layer in range(2, self.num_layers + 1):
            current_outputs = []
            # 遍历当前层的分支
            for branch_idx, branch in enumerate(self.branches[f"layer_{layer}"]):
                input_idx = branch_idx // 2  # 每两个分支共享同一个上一层的输出
                branch_input = prev_outputs[input_idx]
                branch_output = F.silu(branch(branch_input))
                current_outputs.append(branch_output)
            prev_outputs = current_outputs  # 更新上一层输出

        # 拼接所有分支的输出
        x3 = torch.cat(prev_outputs, dim=1)

        # 输出层
        x = self.fc_output(x3)

        return x


# 定义第一个偶函数网络
class Even_Subnet1(torch.nn.Module):
    def __init__(self, num_neurons=32):
        super(Even_Subnet1, self).__init__()
        self.fc1 = torch.nn.Linear(1, num_neurons, bias=False)
        self.fc2 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc3 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc3_2 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc3_3 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc3_4 = torch.nn.Linear(num_neurons, num_neurons, bias=False)

        # 输出层
        self.bias_only = torch.nn.Parameter(torch.zeros(1))  # 第一个神经元只有偏置
        self.fc4 = torch.nn.Linear(num_neurons * 4, 1, bias=False)

        self.apply(initialize_weight)

    def forward(self, x):
        x = torch.exp(-(self.fc1(x) ** 2))
        x = (self.fc2(x) ** 2) * torch.exp(-(self.fc2(x) ** 2))
        x_1 = (self.fc3(x) ** 2) * torch.exp(-(self.fc3(x)) ** 2)
        x_2 = torch.tanh(- (self.fc3_2(x)) ** 2)
        x_3 = torch.exp(-(self.fc3_3(x)) ** 2)
        x_4 = (self.fc3_4(x)) ** 2 / (1 + (self.fc3_4(x)) ** 2)
        x = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        # x = self.fc4(x) + self.bias_only
        x = self.fc4(x)
        return x


# 定义第二个偶函数网络
class Even_Subnet2(torch.nn.Module):
    def __init__(self, num_neurons=32):
        super(Even_Subnet2, self).__init__()
        self.fc1 = torch.nn.Linear(1, num_neurons, bias=False)
        self.fc2 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc2_2 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc2_3 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc2_4 = torch.nn.Linear(num_neurons, num_neurons, bias=False)

        # 输出层
        self.bias_only = torch.nn.Parameter(torch.zeros(1))  # 第一个神经元只有偏置
        self.fc3 = torch.nn.Linear(num_neurons * 4, 1, bias=False)

        self.apply(initialize_weight)

    def forward(self, x):
        x = (self.fc1(x) ** 2) / (1 + (self.fc1(x)) ** 4)
        x_1 = (self.fc2(x) ** 2) / (1 + (self.fc2(x)) ** 2)
        x_2 = torch.sin(self.fc2_2(x) ** 2) / (1 + (self.fc2_2(x)) ** 2)
        x_3 = torch.exp(-(self.fc2_3(x)) ** 2)
        x_4 = (self.fc2_4(x)) ** 2 / (1 + (self.fc2_4(x)) ** 4)
        x = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        x = self.fc3(x) + self.bias_only
        return x


# 定义第奇函数网络
class Odd_Subnet(torch.nn.Module):
    def __init__(self, num_neurons=32):
        super(Odd_Subnet, self).__init__()

        self.fc1 = torch.nn.Linear(1, num_neurons, bias=False)
        self.fc2 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc3 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc3_2 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc3_3 = torch.nn.Linear(num_neurons, num_neurons, bias=False)
        self.fc3_4 = torch.nn.Linear(num_neurons, num_neurons, bias=False)

        # 输出层
        self.fc4 = torch.nn.Linear(num_neurons * 4, 1, bias=False)

        self.apply(initialize_weight)

    def forward(self, x):
        # 使用 sin 激活函数
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x_1 = torch.arctan(self.fc3(x))
        x_2 = torch.tanh(self.fc3_2(x))
        x_3 = self.fc3_3(x) * torch.exp(-(self.fc3_3(x) ** 2))
        x_4 = (self.fc3_4(x)) / (1 + (self.fc3_4(x) ** 2))
        x = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        x = self.fc4(x)
        return x


# 定义第二个子网络
class Subnet2(torch.nn.Module):
    def __init__(self, num_neurons=32):
        super(Subnet2, self).__init__()
        self.even_subnet1 = Even_Subnet1(num_neurons=num_neurons)
        self.even_subnet2 = Even_Subnet2(num_neurons=num_neurons)

    def forward(self, x):
        x_1 = self.even_subnet1(x)
        # return x_1

        x_2 = self.even_subnet2(x)
        return x_1 * x_2


# 定义第三个子网络
class Subnet3(torch.nn.Module):
    def __init__(self, num_neurons=32):
        super(Subnet3, self).__init__()

        self.odd_subnet = Odd_Subnet(num_neurons=num_neurons)
        self.even_subnet = Even_Subnet2(num_neurons=num_neurons)

    def forward(self, x):
        x_1 = self.odd_subnet(x)
        # return x_1

        x_2 = self.even_subnet(x)
        return x_1 * x_2


# 定义不同的网络类型，用于测试
class Subnet4(torch.nn.Module):
    def __init__(self, num_neurons=32):
        super(Subnet4, self).__init__()
        self.fc1 = torch.nn.Linear(1, num_neurons)
        self.fc2 = torch.nn.Linear(num_neurons, num_neurons)
        # self.fc3 = torch.nn.Linear(num_neurons, num_neurons)
        # self.fc4 = torch.nn.Linear(num_neurons, num_neurons)
        self.fc_outut = torch.nn.Linear(num_neurons, 1)

        # self.apply(glorot_uniform_init)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        # x = F.gelu(self.fc3(x))
        # x = F.gelu(self.fc4(x))

        x = self.fc_outut(x)  # 输出层无激活函数
        return x


class Subnet4_2(torch.nn.Module):
    def __init__(self, num_neurons=32):
        super(Subnet4_2, self).__init__()
        self.fc1 = torch.nn.Linear(1, num_neurons)
        self.fc2_1 = torch.nn.Linear(num_neurons, num_neurons // 2)
        self.fc2_2 = torch.nn.Linear(num_neurons, num_neurons // 2)
        self.fc3_1 = torch.nn.Linear(num_neurons // 2, num_neurons // 2 ** 2)
        self.fc3_2 = torch.nn.Linear(num_neurons // 2, num_neurons // 2 ** 2)
        self.fc3_3 = torch.nn.Linear(num_neurons // 2, num_neurons // 2 ** 2)
        self.fc3_4 = torch.nn.Linear(num_neurons // 2, num_neurons // 2 ** 2)
        self.fc4 = torch.nn.Linear(num_neurons, num_neurons)
        self.fc_outut = torch.nn.Linear(num_neurons, 1)

        # self.apply(glorot_uniform_init)

    def forward(self, x):
        x1 = F.gelu(self.fc1(x))

        x2_1 = F.gelu(self.fc2_1(x1))
        x2_2 = F.gelu(self.fc2_2(x1))

        x3_1 = F.gelu(self.fc3_1(x2_1))
        x3_2 = F.gelu(self.fc3_2(x2_1))
        x3_3 = F.gelu(self.fc3_3(x2_2))
        x3_4 = F.gelu(self.fc3_4(x2_2))

        x3 = torch.cat((x3_1, x3_2, x3_3, x3_4), dim=1)
        # x = F.gelu(self.fc3(x))
        # x = F.gelu(self.fc4(x))

        x = self.fc_outut(x3)  # 输出层无激活函数
        return x


# 定义不同的网络类型，用于测试
class Subnet5(torch.nn.Module):
    def __init__(self, num_neurons=32, num_layers=5):
        super(Subnet5, self).__init__()
        # self.subnet = Subnet4_2(num_neurons=num_neurons)
        self.subnet = BS_net(num_neurons=num_neurons, num_layers=num_layers)

        self.initialize_weights()

    def forward(self, x):
        positive_output = self.subnet(x)
        negative_output = self.subnet(-x)

        x = 0.5 * (positive_output + negative_output)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):  # 检查模块是否是 nn.Linear
                # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.xavier_normal_(m.weight, gain=1)  # Glorot Uniform 初始化
                # if self.bias is not None:
                #     nn.init.normal_(self.bias, std=0.2)  # 偏置初始化为 0
                #
                # if self.bias is not None:
                #     nn.init.xavier_uniform_(self.bias)
                # print("initialize weight of Subnet5")
                if m.bias is not None:
                    # nn.init.normal_(m.bias, std=0.2)
                    # nn.init.uniform_(m.bias, a=-1, b=1)
                    nn.init.zeros_(m.bias)
                # if m.bias is not None:
                #     fan_in = m.weight.size(1)  # 输入特征数量
                #     bound = 1 / math.sqrt(fan_in)
                #     nn.init.uniform_(m.bias, a=-bound, b=bound)


class Subnet6(torch.nn.Module):
    def __init__(self, num_neurons=32, num_layers=5):
        super(Subnet6, self).__init__()
        # self.subnet = Subnet4(num_neurons=num_neurons)
        self.subnet = BS_net(num_neurons=num_neurons, num_layers=num_layers)
        self.initialize_weights()

    def forward(self, x):
        positive_output = self.subnet(x)
        negative_output = self.subnet(-x)

        x = 0.5 * (positive_output - negative_output)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):  # 检查模块是否是 nn.Linear
                # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Glorot Uniform 初始化
                # if self.bias is not None:
                #     nn.init.normal_(self.bias, std=0.2)  # 偏置初始化为 0
                #
                # if self.bias is not None:
                #     nn.init.xavier_uniform_(self.bias)
                # print("initialize weight of Subnet6")
                if m.bias is not None:
                    # nn.init.normal_(m.bias, std=0.2)
                    nn.init.zeros_(m.bias)
                # if self.bias is not None:
                #     fan_in = self.weight.size(1)  # 输入特征数量
                #     bound = 1 / math.sqrt(fan_in)
                #     nn.init.uniform_(self.bias, a=-bound, b=bound)


# 定义主网络
class MainNet(torch.nn.Module):
    def __init__(self, subnet1_neurons=2 ** 4
                 , subnet2_neurons=2 ** 4, subnet3_neurons=2 ** 4):
        super(MainNet, self).__init__()
        # self.subnet1 = Subnet1(num_neurons=subnet1_neurons)
        self.subnet1 = BS_net(num_neurons=subnet1_neurons, input_num=2, output_num=2, num_layers=3)

        # self.even_subnet = Subnet2(num_neurons=subnet2_neurons)
        self.subnet2 = Subnet5(num_neurons=subnet2_neurons, num_layers=3)
        self.subnet3 = Subnet6(num_neurons=subnet3_neurons, num_layers=3)

        # 将 regularizer 定义为列表，符合 DeepXDE 的要求
        # self.regularizer = ["l2", 1e-5]  # "l2" 表示 L2 正则化，1e-5 是正则化强度
        # self.regularizer = None
        # self._initialize_weights()

    def forward(self, inputs):
        # 检查是否为 Tensor
        if isinstance(inputs, torch.Tensor):
            x, z = inputs[:, 0:1], inputs[:, 1:2]
        else:
            raise TypeError(f"Expected Tensor, but got {type(inputs)}")

        # 子网络输入
        input_subnet1 = inputs
        input_subnet2 = x
        input_subnet3 = x

        # 子网络输出
        out1 = self.subnet1(input_subnet1)  # 子网1输出 [p, q]

        out2 = self.subnet2(input_subnet2)  # 子网2输出 [v]
        # out2 = self.even_subnet(input_subnet2)
        out3 = self.subnet3(input_subnet3)  # 子网3输出 [w]

        # 拼接输出 [p, q, v, w]
        return torch.cat([out1, out2, out3], dim=1)

    def custom_regularize(self):
        """
        DeepXDE 会自动调用该方法计算正则化损失
        """
        print("custom_regularize method is called!")
        # 从 regularizer 中获取正则化类型和强度
        if self.regularizer[0] != "l2":
            raise ValueError("Only L2 regularization is supported.")
        alpha = self.regularizer[1]

        # 计算所有可训练参数的 L2 正则化
        l2_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.sum(param ** 2)
        return alpha * l2_loss


# 定义 DeepXDE 模型
def main():
    net = MainNet()
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型的总参数数量: {total_params}")
    test_input = torch.rand(10, 2)  # 创建形状为 (10, 2) 的随机输入
    output = net(test_input)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()

    bs_net = BS_net(num_neurons=2 ** 4)
    print(bs_net)
    # 随机输入数据 (batch_size=8, 输入维度=1)
    x = torch.randn(8, 1)

    # 前向传播
    output = bs_net(x)
