import torch
import math
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn


# 定义 Glorot Uniform 初始化函数
def initialize_weight(m):
    if isinstance(m, (nn.Linear, ZeroTrainLinear)):
        # nn.init.kaiming_uniform_(m.weight)
        # nn.init.xavier_normal_(m.weight, gain=1)
        nn.init.xavier_uniform_(m.weight, gain=1)

        # print("initialized weight")

        if m.bias is not None:
            # nn.init.normal_(m.bias, std=0.2)
            nn.init.zeros_(m.bias)
            # nn.init.constant_(m.bias, 0.01)

        # if m.bias is not None:
        #     nn.init.xavier_uniform_(m.bias)

        # if m.bias is not None:
        #     fan_in = m.weight.size(1)  # 输入特征数量
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(m.bias, a=-bound, b=bound)


class ZeroTrainLinear(nn.Module):
    def __init__(self, input_dim, output_dim, mask=None):
        super().__init__()
        # 可训练的权重和偏置
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

        if mask is not None:
            # 把 mask 存成 float 类型，并注册为 buffer
            mask = mask.to(torch.float32)
            self.register_buffer('mask', mask)

            # 一开始就把对应位置的 weight 置 0
            with torch.no_grad():
                self.weight.mul_(self.mask)

            # 注册一个 backward hook：在反向传播时，把 weight.grad 中
            # 对应 mask=0 的位置也置为 0
            # self.weight.register_hook(lambda grad: grad * self.mask)
        else:
            self.mask = None

    def forward(self, x):
        # 如果有 mask，就在前向再次乘一下（可选，因为我们在 __init__ 已经清零了）
        w = self.weight * self.mask if self.mask is not None else self.weight
        return F.linear(x, w, self.bias)


class BS_net(nn.Module):
    def __init__(self, input_num=1, output_num=1, num_neurons=32, num_layers=3):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_layers = num_layers

        # 第一层
        self.fc1 = nn.Linear(input_num, num_neurons)

        # 隐藏层（num_layers-1 层）
        self.layers = nn.ModuleList()
        for layer in range(2, num_layers + 1):
            mask = self._create_mask(layer)
            self.layers.append(ZeroTrainLinear(num_neurons, num_neurons, mask=mask))

        # 输出层
        self.fc_out = nn.Linear(num_neurons, output_num)

        # 初始化权重
        self.apply(initialize_weight)

    def _create_mask(self, layer):
        """
        为第 layer 层生成一个 [num_neurons, num_neurons] 的 mask：
        - 层有 2^(layer-1) 个分支，每个分支宽度 w_out = num_neurons // 2^(layer-1)
        - 它的父分支数 = 2^(layer-2)，父分支宽度 w_in = num_neurons // 2^(layer-2)
        - 对于输出神经元 j，算出它属于哪个分支 b_out = j // w_out，
          然后它的父分支索引 pb = b_out // 2。
        - 只有 input 索引 i 落在 [pb*w_in, (pb+1)*w_in) 时，mask[j,i]=1，其余 =0。
        """
        B_out = 2 ** (layer - 1)
        w_out = self.num_neurons // B_out
        B_in = 2 ** (layer - 2)
        w_in = self.num_neurons // B_in

        mask = torch.zeros((self.num_neurons, self.num_neurons), dtype=torch.float32)
        for j in range(self.num_neurons):
            b_out = j // w_out
            pb = b_out // 2
            start = pb * w_in
            end = start + w_in
            mask[j, start:end] = 1
        return mask

    def forward(self, x):
        x = F.silu(self.fc1(x))
        for layer in self.layers:
            x = F.silu(layer(x))
        return self.fc_out(x)


class Subnet5(torch.nn.Module):
    def __init__(self, num_neurons=32, num_layers=3):
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
            if isinstance(m, (nn.Linear, ZeroTrainLinear)):
                # nn.init.xavier_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=1)

                # if self.bias is not None:
                #     nn.init.normal_(self.bias, std=0.2)  # 偏置初始化为 0
                #
                # if self.bias is not None:
                #     nn.init.xavier_uniform_(self.bias)

                if m.bias is not None:
                    # nn.init.normal_(m.bias, std=0.2)
                    # nn.init.uniform_(m.bias, -1, 1)
                    nn.init.zeros_(m.bias)
                    # nn.init.constant_(m.bias, 0.01)
        # print("initialize weight of Subnet5")


class Subnet6(torch.nn.Module):
    def __init__(self, num_neurons=32, num_layers=3):
        super(Subnet6, self).__init__()
        # self.subnet = Subnet4_2(num_neurons=num_neurons)
        self.subnet = BS_net(num_neurons=num_neurons, num_layers=num_layers)
        self.initialize_weights()

    def forward(self, x):
        positive_output = self.subnet(x)
        negative_output = self.subnet(-x)

        x = 0.5 * (positive_output - negative_output)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, ZeroTrainLinear)):
                # nn.init.xavier_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=0.5)

                # if self.bias is not None:
                #     nn.init.normal_(self.bias, std=0.2)  # 偏置初始化为 0
                #
                # if self.bias is not None:
                #     nn.init.xavier_uniform_(self.bias)

                if m.bias is not None:
                    # nn.init.normal_(m.bias, std=0.2)
                    # nn.init.uniform_(m.bias, -0.1, 0.1)
                    nn.init.zeros_(m.bias)
                    # nn.init.constant_(m.bias, 0.01)
        # print("initialize weight of Subnet6")


# 定义主网络
class MainNet(torch.nn.Module):
    def __init__(self, subnet1_neurons=2 ** 6, subnet2_neurons=2 ** 5, subnet3_neurons=2 ** 5):
        super(MainNet, self).__init__()
        # self.subnet1 = Subnet1(num_neurons=subnet1_neurons)
        self.subnet1 = BS_net(num_neurons=subnet1_neurons, input_num=2, output_num=2)

        # self.even_subnet = Subnet2(num_neurons=subnet2_neurons)
        self.subnet2 = Subnet5(num_neurons=subnet2_neurons)
        self.subnet3 = Subnet6(num_neurons=subnet3_neurons)

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


# 定义 DeepXDE 模型
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
