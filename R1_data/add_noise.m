% 示例矩阵
data = u_exact_imag;

% 噪声水平（可以调整）
noise_level = 0.05; 

% 生成正负随机方向矩阵（1或-1）
% 生成正负随机方向矩阵（1或-1）
direction = 2 * randi([0, 1], size(data)) - 1; % 生成-1或1的矩阵，确保每个数据有随机方向

% 生成噪声矩阵
noise = data * noise_level .* direction;

% 添加噪声
data_noisy = data + noise;

% 输出结果
disp('原矩阵：');
disp(data);

disp('添加噪声后的矩阵：');
disp(data_noisy);
