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
        nn.init.xavier_uniform_(m.weight, gain=0.5)

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


# class BS_net(torch.nn.Module):
#     def __init__(self, input_num=1, output_num=1, num_neurons=32, num_layers=4):
#         super(BS_net, self).__init__()
#         self.num_neurons = num_neurons
#         self.num_layers = num_layers
#         self.input_num = input_num
#         self.output_num = output_num
#
#         # 定义第一层
#         self.fc1 = torch.nn.Linear(input_num, num_neurons)
#
#         # 定义隐藏层分支（从第二层开始）
#         self.branches = torch.nn.ModuleDict()
#         for layer in range(2, num_layers + 1):  # 层数从2开始，到num_layers结束
#             num_branches = 2 ** (layer - 1)  # 当前层的分支数量是2的(layer-1)次方
#             input_dim = num_neurons // (2 ** (layer - 2))  # 输入维度：前一层的输出
#             output_dim = num_neurons // (2 ** (layer - 1))  # 输出维度：当前分支的输出
#             # 为当前层的每个分支创建Linear层
#             self.branches[f"layer_{layer}"] = torch.nn.ModuleList([
#                 torch.nn.Linear(input_dim, output_dim) for _ in range(num_branches)
#             ])
#
#         # 定义输出层
#         self.fc_output = torch.nn.Linear(num_neurons, output_num)
#
#         # 初始化权重
#         self.apply(initialize_weight)
#
#     def forward(self, x):
#         # 第一层处理
#         x1 = torch.tanh(self.fc1(x))
#
#         # 初始化上一层的输出为列表
#         prev_outputs = [x1]
#
#         # 遍历隐藏层
#         for layer in range(2, self.num_layers + 1):
#             current_outputs = []
#             # 遍历当前层的分支
#             for branch_idx, branch in enumerate(self.branches[f"layer_{layer}"]):
#                 input_idx = branch_idx // 2  # 每两个分支共享同一个上一层的输出
#                 branch_input = prev_outputs[input_idx]
#                 branch_output = torch.tanh(branch(branch_input))
#                 current_outputs.append(branch_output)
#             prev_outputs = current_outputs  # 更新上一层输出
#
#         # 拼接所有分支的输出
#         x3 = torch.cat(prev_outputs, dim=1)
#
#         # 输出层
#         x = self.fc_output(x3)
#
#         return


class ZeroTrainLinear(nn.Module):
    def __init__(self, input_dim, output_dim, mask=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        if mask is not None:
            self.register_buffer('mask', mask)
            # 一开始就把不训练的权重置零
            with torch.no_grad():
                self.weight.mul_(self.mask)
        else:
            self.mask = None

    def forward(self, x):
        w = self.weight * self.mask if self.mask is not None else self.weight
        return F.linear(x, w, self.bias)


class BS_net(nn.Module):
    def __init__(self, input_num=1, output_num=1, num_neurons=32, num_layers=3):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_layers = num_layers

        # 第一层
        self.fc1 = ZeroTrainLinear(input_num, num_neurons)

        # 隐藏层（num_layers-1 层）
        self.layers = nn.ModuleList()
        for layer in range(2, num_layers + 1):
            mask = self._create_mask(layer)
            self.layers.append(ZeroTrainLinear(num_neurons, num_neurons, mask=mask))

        # 输出层
        self.fc_out = ZeroTrainLinear(num_neurons, output_num)

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
            if isinstance(m, nn.Linear):  # 检查模块是否是 nn.Linear
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
                    # nn.init.zeros_(m.bias)
                    nn.init.constant_(m.bias, 0.01)
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
            if isinstance(m, nn.Linear):  # 检查模块是否是 nn.Linear
                # nn.init.xavier_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=1)

                # if self.bias is not None:
                #     nn.init.normal_(self.bias, std=0.2)  # 偏置初始化为 0
                #
                # if self.bias is not None:
                #     nn.init.xavier_uniform_(self.bias)

                if m.bias is not None:
                    # nn.init.normal_(m.bias, std=0.2)
                    # nn.init.uniform_(m.bias, -0.1, 0.1)
                    # nn.init.zeros_(m.bias)
                    nn.init.constant_(m.bias, 0.01)
        # print("initialize weight of Subnet6")


# 定义主网络
class MainNet(torch.nn.Module):
    def __init__(self, subnet1_neurons=2 ** 4, subnet2_neurons=2 ** 4, subnet3_neurons=2 ** 4):
        super(MainNet, self).__init__()
        # self.subnet1 = Subnet1(num_neurons=subnet1_neurons)
        self.subnet1 = BS_net(num_neurons=subnet1_neurons, input_num=2, output_num=2)

        # self.even_subnet = Subnet2(num_neurons=subnet2_neurons)
        self.subnet2 = Subnet5(num_neurons=subnet2_neurons)
        self.subnet3 = Subnet6(num_neurons=subnet3_neurons)

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
    print(f"Output shape: {output.shape} \n")

    print(f"Model structure: {net}\n\n")


if __name__ == "__main__":
    main()
