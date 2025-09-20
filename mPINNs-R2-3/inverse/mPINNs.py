import torch
import torch.nn as nn

#激活函数
activation_functions = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU  # 添加 SiLU
}


class MainNet(nn.Module):
    def __init__(self, input_size=2, output_size=4, hidden_layers=3, hidden_size=128, activation="silu"):
        super(MainNet, self).__init__()

        if activation.lower() not in activation_functions:
            raise ValueError(f"Unsupported activation function: {activation}")

        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation_functions[activation.lower()]())

        # 中间隐藏层
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_functions[activation.lower()]())

        # 输出层
        layers.append(nn.Linear(hidden_size, output_size))

        # 构建模型
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def main():
    net = MainNet()
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型的总参数数量: {total_params}")

    test_input = torch.rand(10, 2)  # 创建形状为 (10, 2) 的随机输入
    output = net(test_input)
    print(f"Output shape: {output.shape} \n")

    print(f"Model structure: {net}\n\n")


if __name__ == "__main__":
    main()
