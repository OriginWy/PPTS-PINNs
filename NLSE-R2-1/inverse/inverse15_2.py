import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

from inverse_pfnn15_2 import MainNet
from torch.autograd import grad
import pandas as pd
import os

# 常数值
A = 1.0
v0 = 1.0
w0 = 0.2
sigma = -1.0


# 计算实际函数值
def u_actual(x, t):
    # Compute omega and miu
    omega = (1 / 4) * (np.sqrt(sigma ** 2 + 4 * v0) - sigma)
    miu = 2 * omega + sigma * np.log(A ** 2) + ((w0 ** 2) / (16 * omega))

    # Compute phi as a function of x and t
    phi = A * np.exp(-omega * x ** 2) * np.exp(-1j * ((w0 * x) / (4 * omega) + miu * t))

    return phi


# 定义 V_pt 函数
def V_pt(x):
    return v0 * x ** 2 + 1j * w0 * x


# 加载数据
u_exact_real_data = scipy.io.loadmat('../data/u_exact_real1.mat')['u_exact_real']
u_exact_imag_data = scipy.io.loadmat('../data/u_exact_imag1.mat')['u_exact_imag']

# 将数据转换为 PyTorch 张量，并移动到 GPU（如果可用）
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
u_exact_real = torch.tensor(u_exact_real_data.reshape((-1, 1)), dtype=torch.float32).to(device)
u_exact_imag = torch.tensor(u_exact_imag_data.reshape((-1, 1)), dtype=torch.float32).to(device)

# 假设以下为网格点信息
x_lower, x_upper = -1.0, 1.0  # 根据实际情况修改
z_lower, z_upper = 0, 1.0  # 根据实际情况修改
num_z, num_x = u_exact_real_data.shape  # 网格划分数量

# 创建网格点
x = np.linspace(x_lower, x_upper, num_x)
z = np.linspace(z_lower, z_upper, num_z)
xx, zz = np.meshgrid(x, z, indexing='xy')  # 创建网格点
xx = torch.tensor(xx.reshape((-1, 1)), dtype=torch.float32, device=device)
zz = torch.tensor(zz.reshape((-1, 1)), dtype=torch.float32, device=device)

# 展平后合并
input_xz = torch.cat((xx, zz), dim=1)  # [num_points, 2]
input_xz.requires_grad_(True)


# 内部观测数据损失函数定义
def loss_u_actual(net, input_xz, u_exact_real, u_exact_imag):
    """
    计算神经网络输出和实际函数 u_actual 的损失。

    参数:
    - net: 定义的神经网络 (MainNet)
    - input_xz: 输入网格点 [x, z]，形状 [num_points, 2]
    - u_exact_real: 真实 u_actual 实部的值
    - u_exact_imag: 真实 u_actual 虚部的值

    返回:
    - 计算的 MSE 损失值
    """
    # 网络预测
    output = net(input_xz)  # 取 [x], [z] 作为输入
    p = output[:, 0:1]
    q = output[:, 1:2]

    # 定义损失
    mse_loss = nn.MSELoss()
    loss_real = mse_loss(p, u_exact_real)  # 实部损失
    loss_imag = mse_loss(q, u_exact_imag)  # 虚部损失

    return loss_real + loss_imag


num_z = 201  # z 网格点数量

# 网格划分
z = np.linspace(z_lower, z_upper, num_z).reshape((-1, 1))

# 边界数据生成
u_actual_lower = u_actual(x_lower, z)
u_actual_upper = u_actual(x_upper, z)
# 提取实部和虚部
u_lower_real = torch.tensor(u_actual_lower.real, dtype=torch.float32).to(device)
u_lower_imag = torch.tensor(u_actual_lower.imag, dtype=torch.float32).to(device)
u_upper_real = torch.tensor(u_actual_upper.real, dtype=torch.float32).to(device)
u_upper_imag = torch.tensor(u_actual_upper.imag, dtype=torch.float32).to(device)

# 读取.mat文件
data = scipy.io.loadmat('../data/pt_left1.mat')
pt_left = data['pt_left'][0, 0]
real_part = np.real(pt_left)
left_real_part = real_part
right_real_part = real_part
left_imag_part = np.imag(pt_left)
right_imag_part = -left_imag_part
z_input = torch.tensor(z, dtype=torch.float32).to(device)  # 边界输入中的z


# 边界损失函数定义
def boundary_loss(net, z_input, u_lower_real, u_lower_imag, u_upper_real, u_upper_imag,
                  left_real_part, left_imag_part, right_real_part, right_imag_part):
    """
    定义边界损失函数。

    参数:
    - net: 神经网络
    - z: 边界上的 z 坐标 (torch.Tensor)
    - u_lower_real, u_lower_imag: u_actual 在 x_lower 边界的真实值
    - u_upper_real, u_upper_imag: u_actual 在 x_upper 边界的真实值
    - left_real_part, left_imag_part: V_pt 在 x_lower 的真实值
    - right_real_part, right_imag_part: V_pt 在 x_upper 的真实值

    返回:
    - 总损失 (torch.Tensor)
    """
    # 生成 z 输入
    # z_input = z_input.reshape(-1).to(device)  # [num_z, 1]

    # 边界输入
    x_lower_tensor = torch.full_like(z_input, x_lower)
    x_upper_tensor = torch.full_like(z_input, x_upper)
    input_left = torch.cat((x_lower_tensor, z_input), dim=1)
    input_right = torch.cat((x_upper_tensor, z_input), dim=1)

    # 网络输出
    output_left = net(input_left)
    output_right = net(input_right)
    p_lower = output_left[:, 0:1]
    q_lower = output_left[:, 1:2]
    v_left = output_left[:, 2:3]
    w_left = output_left[:, 3:4]

    p_upper = output_right[:, 0:1]
    q_upper = output_right[:, 1:2]
    v_right = output_right[:, 2:3]
    w_right = output_right[:, 3:4]

    # 损失计算 (MSE)
    mse_loss = nn.MSELoss()

    # u_actual 边界损失
    loss_lower_real = mse_loss(p_lower, u_lower_real)
    loss_lower_imag = mse_loss(q_lower, u_lower_imag)
    loss_upper_real = mse_loss(p_upper, u_upper_real)
    loss_upper_imag = mse_loss(q_upper, u_upper_imag)

    # V_pt 边界损失
    loss_v_left_real = mse_loss(v_left, torch.full_like(v_left, left_real_part))
    loss_v_left_imag = mse_loss(w_left, torch.full_like(w_left, left_imag_part))
    loss_v_right_real = mse_loss(v_right, torch.full_like(v_right, right_real_part))
    loss_v_right_imag = mse_loss(w_right, torch.full_like(w_right, right_imag_part))

    # 总损失
    total_loss = (loss_lower_real + loss_lower_imag +
                  loss_upper_real + loss_upper_imag +
                  loss_v_left_real + loss_v_left_imag +
                  loss_v_right_real + loss_v_right_imag)
    return total_loss


# 生成 x 和 z 数据
num_points = 401

x_boundary = np.linspace(x_lower, x_upper, num_points).reshape((-1, 1))
z_boundary = np.zeros_like(x_boundary)  # 固定 z = 0

# 使用 u_actual 计算边界上的精确解
u_boundary_complex = u_actual(x_boundary, z_boundary)

# 提取实部和虚部
u_boundary_real = np.real(u_boundary_complex)
u_boundary_imag = np.imag(u_boundary_complex)

# 转换为 Torch 张量并移动到 GPU（如果可用）
x_boundary_tensor = torch.tensor(x_boundary, dtype=torch.float32).to(device)
u_boundary_real_tensor = torch.tensor(u_boundary_real, dtype=torch.float32).to(device)
u_boundary_imag_tensor = torch.tensor(u_boundary_imag, dtype=torch.float32).to(device)


# 定义边界损失函数
def initial_loss(model, x_boundary, u_boundary_real, u_boundary_imag):
    """
    计算边界损失项。

    Parameters:
        model (torch.nn.Module): MainNet 模型
        x_boundary (torch.Tensor): 边界上的输入坐标 (x, z=0)
        u_boundary (torch.Tensor): 精确解的实部和虚部

    Returns:
        torch.Tensor: 边界损失
    """
    # 扩展 x_boundary 为 [x, z] 格式，z 固定为 0
    z_boundary = torch.zeros_like(x_boundary).to(device)
    xz_boundary = torch.cat([x_boundary, z_boundary], dim=1)

    # 通过网络模型预测 u 的实部和虚部
    u_pred = model(xz_boundary)  # 输出 [p, q, v, w]
    u_pred_real = u_pred[:, 0:1]  # p，对应 u_actual 的实部
    u_pred_imag = u_pred[:, 1:2]  # q，对应 u_actual 的虚部

    # 计算预测值与精确值之间的均方误差
    loss_real = torch.mean((u_pred_real - u_boundary_real) ** 2)
    loss_imag = torch.mean((u_pred_imag - u_boundary_imag) ** 2)

    # 返回总的边界损失
    return loss_real + loss_imag


# Helper function to calculate gradients
def compute_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad_vals = grad(
        y, x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return grad_vals


# Helper function to calculate Hessian
def compute_hessian(y, x, component_idx, i, j):
    grad_y = compute_gradient(y[:, component_idx: component_idx + 1], x)
    hessian = compute_gradient(grad_y[:, i: i + 1], x)[:, j:j + 1]
    return hessian


# Define the PDE loss and gradients loss in PyTorch
def pde_losses(x, u):
    u_real = u[:, 0:1]  # Real part
    u_imag = u[:, 1:2]  # Imaginary part
    V = u[:, 2:3]  # V part, real part of $V_{pt}$
    W = u[:, 3:4]  # W part, imaginary part of $V_{pt}$

    # Compute partial derivatives
    u_real_z = compute_gradient(u_real, x)[:, 1:2]  # ∂u_real/∂z
    u_imag_z = compute_gradient(u_imag, x)[:, 1:2]  # ∂u_imag/∂z

    u_real_xx = compute_hessian(u, x, component_idx=0, i=0, j=0)  # ∂²u_real/∂x²
    u_imag_xx = compute_hessian(u, x, component_idx=1, i=0, j=0)  # ∂²u_imag/∂x²

    # Define PDE equations
    pde_real = -u_imag_z + u_real_xx - V * u_real + W * u_imag + torch.log(
        torch.clamp(u_real ** 2 + u_imag ** 2, 1e-20)) * u_real
    pde_imag = u_real_z + u_imag_xx - V * u_imag - W * u_real + torch.log(
        torch.clamp(u_real ** 2 + u_imag ** 2, 1e-20)) * u_imag

    # Compute PDE loss (mean squared error of the PDE residuals)
    pde_loss = torch.mean(pde_real ** 2) + torch.mean(pde_imag ** 2)

    # Add gradients of the PDE
    # pde_real_x = compute_gradient(pde_real, x)  # ∂pde_real/∂x and  ∂pde_real/∂z
    # # pde_real_z = compute_gradient(pde_real, x)[:, 1:2]  # ∂pde_real/∂z
    # pde_imag_x = compute_gradient(pde_imag, x)  # ∂pde_imag/∂x and ∂pde_imag/∂z
    # # pde_imag_z = compute_gradient(pde_imag, x)[:, 1:2]  # ∂pde_imag/∂z
    #
    # # Compute gradients loss (mean squared error of the gradients)
    # gradients_loss = (
    #         torch.mean(pde_real_x ** 2)
    #         # + torch.mean(pde_real_z ** 2)
    #         + torch.mean(pde_imag_x ** 2)
    #     # + torch.mean(pde_imag_z ** 2)
    # )
    #
    # return pde_loss + gradients_loss
    return pde_loss


# 定义封装的测试和绘图函数
def test_and_plot(model, epoch):
    """
    测试模型并绘制结果对比图
    :param model: 训练的模型
    :param V_pt: 目标函数 V_pt(x)
    :param x_lower: x 的下界
    :param x_upper: x 的上界
    :param device: 设备 (CPU 或 GPU)
    :param epoch: 当前的训练轮数
    """
    x_test = np.linspace(x_lower, x_upper, 100)
    z_test = np.zeros_like(x_test)
    test_coords = np.vstack((x_test, z_test)).T
    test_coords_tensor = torch.tensor(test_coords, dtype=torch.float32, device=device)

    # 模型预测
    model.eval()
    with torch.no_grad():
        u_pred = model(test_coords_tensor).cpu().numpy()
    V_pred = u_pred[:, 2]
    W_pred = u_pred[:, 3]

    # 计算真实的 V_pt
    V_pt_values = V_pt(x_test)
    V_pt_real = np.real(V_pt_values)
    V_pt_imag = np.imag(V_pt_values)

    # 绘图
    plt.figure(figsize=(12, 6))

    # V_pred vs V_pt Real Part
    plt.subplot(1, 2, 1)
    plt.plot(x_test, V_pt_real, label='$V_{pt}(x)$ (Real)', linestyle='-', color='blue')
    plt.plot(x_test, V_pred, label='$V_{pred}$ (Real)', linestyle='--', color='orange')
    plt.title(f'inverse15_2 Comparison of $V_{{pred}}$ and $V_{{pt}}(x)$ (Real Part) - Epoch {epoch}')
    plt.xlabel('$x$')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()

    # W_pred vs V_pt Imaginary Part
    plt.subplot(1, 2, 2)
    plt.plot(x_test, V_pt_imag, label='$V_{pt}(x)$ (Imag)', linestyle='-', color='blue')
    plt.plot(x_test, W_pred, label='$W_{pred}$ (Imag)', linestyle='--', color='orange')
    plt.title(f'inverse15_2 Comparison of $W_{{pred}}$ and $V_{{pt}}(x)$ (Imaginary Part) - Epoch {epoch}')
    plt.xlabel('$x$')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    plt.close('all')

# Training loop
# 设置超参数
learning_rate = 10 * 1e-5
num_epochs = 20000
batch_size = 256  # 如果数据过大，可以使用批量训练
log_interval = 1000

# 初始化模型并移动到 GPU
model = MainNet().to(device)
# model = model.double()
# 定义优化器和学习率调度器
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.00)

# 定义余弦退火学习率调度器
# T_max 是余弦退火的周期（单位：epoch），eta_min 是最小学习率
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5*1e-5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

# 初始化 ReduceLROnPlateau 调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # 监控损失值，目标是越小越好
    factor=0.95,  # 学习率缩小的比例
    patience=20,  # 当损失没有改善时，等待 patience 个 epoch 后再调整学习率
    min_lr=1 * 1e-8,
    threshold=1e-4,
)

# 初始化一个空的列表来存储每个 epoch 的损失数据
loss_data = []

test_and_plot(model, epoch=1)
# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式

    # 计算损失函数
    # 内部观测数据损失
    loss_u = loss_u_actual(model, input_xz, u_exact_real, u_exact_imag)

    # 边界损失
    loss_b = boundary_loss(
        model, z_input, u_lower_real, u_lower_imag,
        u_upper_real, u_upper_imag, left_real_part, left_imag_part, right_real_part, right_imag_part
    )

    # 初始损失
    loss_i = initial_loss(model, x_boundary_tensor, u_boundary_real_tensor, u_boundary_imag_tensor)

    # PDE损失
    u_pred = model(input_xz)  # 网络预测

    loss_p = pde_losses(input_xz, u_pred)

    # 总损失
    total_loss = loss_u + loss_b + loss_i + loss_p

    # 反向传播和优化
    optimizer.zero_grad()
    total_loss.backward()

    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # 梯度剪裁
    optimizer.step()

    # 更新学习率
    # if epoch + 1 < 15000:
    # scheduler.step()
    scheduler.step(total_loss.item())

    # 日志打印
    if (epoch + 1) % log_interval == 0 or epoch == 0:
        current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss_u: {loss_u.item():.5e}, "
              f"Loss_b: {loss_b.item():.5e}, "
              f"Loss_i: {loss_i.item():.5e}, "
              f"Loss_p: {loss_p.item():.5e}, "
              f"Total Loss: {total_loss.item():.5e}, "
              f"Learning Rate: {current_lr:.5e}")
        # if (epoch + 1) > 5000:
        # 每 1000 次迭代测试并绘图
        test_and_plot(model, epoch + 1)
        # 保存训练好的模型
        torch.save(model, '../model15/model15_2.pth')

        # 将当前损失值记录到列表中
        loss_data.append({
            'epoch': epoch + 1,
            'Loss_u': loss_u.item(),
            'Loss_b': loss_b.item(),
            'Loss_i': loss_i.item(),
            'Loss_p': loss_p.item(),
            'Total_Loss': total_loss.item()
        })

# 将记录的损失数据转换为 DataFrame
loss_df = pd.DataFrame(loss_data)
# 保存为 CSV 文件
loss_df.to_csv('../data/loss_data.csv', index=False)

# 定义 L-BFGS 优化器
lbfgs_optimizer = optim.LBFGS(model.parameters(), max_iter=500, history_size=100)
num_epochs = 0


# 在循环外定义 closure 函数
def closure():
    lbfgs_optimizer.zero_grad()  # 清除梯度

    # 计算损失函数
    # 内部观测数据损失
    loss_u = loss_u_actual(model, input_xz, u_exact_real, u_exact_imag)

    # 边界损失
    loss_b = boundary_loss(
        model, z_input, u_lower_real, u_lower_imag,
        u_upper_real, u_upper_imag, left_real_part, left_imag_part, right_real_part, right_imag_part
    )

    # 初始损失
    loss_i = initial_loss(model, x_boundary_tensor, u_boundary_real_tensor, u_boundary_imag_tensor)

    # PDE损失
    u_pred = model(input_xz)  # 网络预测
    loss_p = pde_losses(input_xz, u_pred)

    # 总损失
    total_loss = loss_u + loss_b + loss_i + loss_p

    # 反向传播
    total_loss.backward()

    return total_loss


# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式

    # 调用 L-BFGS 优化器进行一步优化
    lbfgs_optimizer.step(closure)

    # 日志打印
    if (epoch + 1) % log_interval == 0:
        # 重新计算损失（用于日志打印）
        with torch.no_grad():
            # 重新计算损失，用于打印
            loss_u = loss_u_actual(model, input_xz, u_exact_real, u_exact_imag)
            loss_b = boundary_loss(
                model, z_input, u_lower_real, u_lower_imag,
                u_upper_real, u_upper_imag, left_real_part, left_imag_part, right_real_part, right_imag_part
            )
            loss_i = initial_loss(model, x_boundary_tensor, u_boundary_real_tensor, u_boundary_imag_tensor)

        input_xz_pde = input_xz.detach().requires_grad_(True)
        u_pred_pde = model(input_xz_pde)
        loss_p = pde_losses(input_xz_pde, u_pred_pde)
        total_loss = loss_u + loss_b + loss_i + loss_p

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss_u: {loss_u.item():.5e}, "
              f"Loss_b: {loss_b.item():.5e}, "
              f"Loss_i: {loss_i.item():.5e}, "
              f"Loss_p: {loss_p.item():.5e}, "
              f"Total Loss: {total_loss.item():.5e}")
        test_and_plot(model, epoch + 1)
        # 保存训练好的模型
        torch.save(model, '../model15/model15_2.pth')

# 保存训练好的模型
torch.save(model, '../model15/model15_2.pth')

print("训练完成，模型已保存为 model15_2.pth。")

# Testing
x_test = np.linspace(x_lower, x_upper, 100)
z_test = np.zeros_like(x_test)
test_coords = np.vstack((x_test, z_test)).T
test_coords_tensor = torch.tensor(test_coords, dtype=torch.float32, device=device)

with torch.no_grad():
    u_pred = model(test_coords_tensor).cpu().numpy()

u_real_pred = u_pred[:, 0]
u_imag_pred = u_pred[:, 1]
V_pred = u_pred[:, 2]
W_pred = u_pred[:, 3]

# First Plot: Predicted u(x, Z)
plt.figure(figsize=(10, 5))
plt.plot(x_test, u_real_pred, label='Real part')
plt.plot(x_test, u_imag_pred, label='Imaginary part')
plt.legend()
plt.title('Predicted $u(x, Z)$')
plt.xlabel('$x$')
plt.ylabel('$u(x, Z)$')
plt.show()

# Second Plot: Comparison of V_pred and V_pt
V_pt_values = V_pt(x_test)
V_pt_real = np.real(V_pt_values)
V_pt_imag = np.imag(V_pt_values)

plt.figure(figsize=(12, 6))

# V_pred vs V_pt Real Part
plt.subplot(1, 2, 1)
plt.plot(x_test, V_pt_real, label='$V_{pt}(x)$ (Real)', linestyle='-', color='blue')
plt.plot(x_test, V_pred, label='$V_{pred}$ (Real)', linestyle='--', color='orange')
plt.title('Comparison of $V_{pred}$ and $V_{pt}(x)$ (Real Part)')
plt.xlabel('$x$')
plt.ylabel('Value')
plt.legend()
plt.grid()

# W_pred vs V_pt Imaginary Part
plt.subplot(1, 2, 2)
plt.plot(x_test, V_pt_imag, label='$V_{pt}(x)$ (Imag)', linestyle='-', color='blue')
plt.plot(x_test, W_pred, label='$W_{pred}$ (Imag)', linestyle='--', color='orange')
plt.title('Comparison of $W_{pred}$ and $V_{pt}(x)$ (Imaginary Part)')
plt.xlabel('$x$')
plt.ylabel('Value')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 绘制模型与PDE精确解之间的误差
# 生成输入数据 (x, z)
x_vals = np.linspace(-x_upper, x_upper, 100)  # 在 [-L, L] 范围内生成 100 个点
z_vals = np.linspace(z_lower, z_upper, 100)  # 在 [0, Z] 范围内生成 100 个点
x_grid, z_grid = np.meshgrid(x_vals, z_vals)  # 创建网格
u_actual_values = u_actual(x_grid, z_grid)  # 实际函数值

# 获取模型输出
# 假设你的 DeepXDE 模型已经训练完毕，可以用 model.predict() 来获取输出
# 需要将网格点重新整理为模型所需的格式
inputs = np.vstack([x_grid.flatten(), z_grid.flatten()]).T
inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
with torch.no_grad():
    u_pred = model(inputs).cpu().numpy()
u_model_values = u_pred[:, 0] + 1j * u_pred[:, 1]  # 提取实部和虚部

# 计算误差
error = np.abs(u_actual_values.flatten() - u_model_values)

# 重新整形误差为网格形状
error_grid = error.reshape(x_grid.shape)

# 绘制误差图
plt.figure(figsize=(8, 6))
cp = plt.contourf(x_grid, z_grid, error_grid, 50, cmap='viridis')
plt.colorbar(cp)
plt.title('inverse15_2 Error between PINNs and actual function $u(x,z)$')
plt.xlabel('x')
plt.ylabel('z')
plt.show()
