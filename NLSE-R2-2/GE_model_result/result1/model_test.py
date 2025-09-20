import numpy
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import pandas as pd

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


x_lower, x_upper = -1.0, 1.0  # 根据实际情况修改
z_lower, z_upper = 0, 1.0  # 根据实际情况修改

model_path = './model15.pth'

# 将数据转换为 PyTorch 张量，并移动到 GPU（如果可用）
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = MainNet().to(device)
model = torch.load(model_path, weights_only=False).to(device)
print(f"Model structure: {model}\n\n")


model.eval()

# plt.ion()
# Testing
x_test = np.linspace(x_lower, x_upper, 101)
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
# plt.figure(figsize=(10, 5))
# plt.plot(x_test, u_real_pred, label='Real part')
# plt.plot(x_test, u_imag_pred, label='Imaginary part')
# plt.legend()
# plt.title('Predicted $u(x, Z)$')
# plt.xlabel('$x$')
# plt.ylabel('$u(x, Z)$')
# plt.show()

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

# Create a DataFrame to store the data
data = {
    'x_test': x_test,
    'V_pt_real': V_pt_real,
    'V_pred': V_pred,
    'V_pt_imag': V_pt_imag,
    'W_pred': W_pred
}

df_pt = pd.DataFrame(data)

# Saving the DataFrame to a CSV file
csv_file_path = './model_predictions.csv'
df_pt.to_csv(csv_file_path, index=False)

# 计算绝对误差
V_real_abs_error = np.abs(V_pred - V_pt_real)
V_imag_abs_error = np.abs(W_pred - V_pt_imag)

# 创建 DataFrame
df_errors = pd.DataFrame({
    'x': x_test,
    'V_abs_error': V_real_abs_error,
    'W_abs_error': V_imag_abs_error
})

# 保存为 CSV 文件
df_errors.to_csv('./prediction_errors.csv', index=False)

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

# Flatten the actual and predicted complex arrays
u_actual_flat = u_actual_values.flatten()
u_model_flat = u_model_values.flatten()

# Infinity norm error
linf_error = np.max(np.abs(u_actual_flat - u_model_flat))

# Relative L2 norm error
l2_error = np.linalg.norm(u_actual_flat - u_model_flat) / np.linalg.norm(u_actual_flat)

print(f"L_inf error of u: {linf_error}")
print(f"Relative L2 error of u: {l2_error}")

# 计算误差
error = np.abs(u_actual_values.flatten() - u_model_values)
print(np.mean(numpy.abs(numpy.real(u_actual_values.flatten() - u_model_values))**2))
# 重新整形误差为网格形状
error_grid = error.reshape(x_grid.shape)

# 绘制误差图
plt.figure(figsize=(8, 6))
cp = plt.contourf(x_grid, z_grid, error_grid, 50, cmap='viridis')
plt.colorbar(cp)
plt.title('inverse Error between PINNs and actual function $u(x,z)$')
plt.xlabel('x')
plt.ylabel('z')
plt.show()

# 创建 DataFrame 用于重整数据
df_u = pd.DataFrame(error_grid, columns=x_vals, index=z_vals)
# 保存为 CSV 文件
csv_file_path = './error_data_for_origin.csv'
df_u.to_csv(csv_file_path)

# Calculate absolute errors
abs_error_V = np.abs(V_pred - V_pt_real)
abs_error_W = np.abs(W_pred - V_pt_imag)

# Calculate relative L2 norm errors
rel_l2_error_V = np.linalg.norm(V_pred - V_pt_real) / np.linalg.norm(V_pt_real)
rel_l2_error_W = np.linalg.norm(W_pred - V_pt_imag) / np.linalg.norm(V_pt_imag)

# Calculate maximum norm errors
max_error_V = np.max(abs_error_V)
max_error_W = np.max(abs_error_W)

# Print the errors
# print(f'Absolute errors:')
# print(f'  V_pred vs V_pt_real: {abs_error_V}')
# print(f'  W_pred vs V_pt_imag: {abs_error_W}')

print(f'Relative L2 norm errors:')
print(f'  V_pred vs V_pt_real: {rel_l2_error_V}')
print(f'  W_pred vs V_pt_imag: {rel_l2_error_W}')

print(f'Maximum norm errors:')
print(f'  V_pred vs V_pt_real: {max_error_V}')
print(f'  W_pred vs V_pt_imag: {max_error_W}')

# Optionally, plot the errors
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(x_test, abs_error_V, label='Absolute error (V_pred vs V_pt_real)')
plt.plot(x_test, abs_error_W, label='Absolute error (W_pred vs V_pt_imag)')
plt.legend()
plt.title('Absolute Errors')
plt.xlabel('$x$')
plt.ylabel('Error')

plt.subplot(2, 1, 2)
plt.plot(x_test, np.abs(V_pred - V_pt_real) / np.abs(V_pt_real), label='Relative error (V_pred vs V_pt_real)')
plt.plot(x_test, np.abs(W_pred - V_pt_imag) / np.abs(V_pt_imag), label='Relative error (W_pred vs V_pt_imag)')
plt.legend()
plt.title('Relative Errors (L2 Norm)')
plt.xlabel('$x$')
plt.ylabel('Relative Error')

plt.tight_layout()

# plt.ioff()
plt.show()
plt.close("all")
