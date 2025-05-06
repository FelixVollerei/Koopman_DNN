% main_run_experiment.m
% 主脚本：配置并运行 Koopman DNN 观测器实验

clearvars; close all; clc;
rng('default'); % for reproducibility

% --- 0. 添加工具路径 ---
addpath('utils');
addpath('models')

% --- 1. 实验配置 ---
% 选择要运行的系统 ('vanderpol', 'duffing', 'lorenz', 'rossler', 'lotka_volterra')
system_choice = 'lorenz'; % <--- 在这里选择系统

% 数据生成选项
force_generate_data = false; % 如果为 true，则始终重新生成数据

% DNN 训练配置
hidden_units = [128, 128, 64]; % DNN 隐藏层结构
validation_split = 0.15;      % 用于验证的数据比例 (0 表示不使用验证集)
max_epochs = 100;             % 最大训练轮数
mini_batch_size = 128;        % 小批量大小
initial_learn_rate = 0.001;   % 初始学习率
learn_rate_drop_period = 25;  % 学习率下降周期 (epochs)
learn_rate_drop_factor = 0.5; % 学习率下降因子
gradient_threshold = 1.0;     % 梯度裁剪阈值 (防止梯度爆炸)
use_gpu = false;              % 是否使用 GPU (如果可用且已配置)

% 观测器配置
test_data_ratio = 0.3; % 用于测试的数据比例 (从生成的数据末尾取)
% 测量矩阵 C: 定义哪些状态是可测量的 (M x D)
% 示例: Lorenz 系统 (D=3)
%   测量 x1 和 x3 -> C = [1 0 0; 0 0 1] (M=2)
%   只测量 x1 -> C = [1 0 0] (M=1)
%   测量所有 -> C = eye(D)
C_map = containers.Map;
C_map('vanderpol') = [1 0]; % 只测量 x1
C_map('duffing') = [1 0];   % 只测量 x1
C_map('lorenz') = [1 0 0; 0 0 1]; % 测量 x1 和 x3
C_map('rossler') = [1 0 0];  % 只测量 x1
C_map('lotka_volterra') = [1 0]; % 只测量 x1 (捕食者)

C = C_map(system_choice);

% 观测器增益 L (D x M)
% 设置为空 [] 或 zeros(D, M) 表示纯预测 (无校正)
% 设计 L 是高级主题，这里我们先用纯预测
L = []; % <--- 设置为非空矩阵以启用校正

% 观测器初始估计 x_hat(0)
% 可以使用真实初始值、加噪声的初始值或零向量
use_true_x0_for_observer = false; % 是否使用真实初始值作为观测器初始值
observer_noise_level = 0.1;     % 如果不使用真实值，给零向量加多大噪声

% --- 2. 检查/生成数据 ---
data_filename = sprintf('%s_data.mat', system_choice);
if ~exist(data_filename, 'file') || force_generate_data
    fprintf('数据文件 %s 不存在或强制重新生成。\n', data_filename);
    generate_data; % 运行生成脚本 (它会生成所有系统的数据)
else
    fprintf('使用已存在的数据文件: %s\n', data_filename);
end

% --- 3. 加载数据并分割测试集 ---
fprintf('加载数据用于训练和测试...\n');
loaded_data = load(data_filename); % <-- 修改：加载到结构体 loaded_data
var_names = fieldnames(loaded_data); % <-- 修改：从结构体获取字段名

% 找到包含 't_' 和 'x_' 的变量名
t_var_index = find(cellfun(@(s) contains(s,'t_'), var_names), 1);
x_var_index = find(cellfun(@(s) contains(s,'x_'), var_names), 1);

if isempty(t_var_index) || isempty(x_var_index)
    error('无法在 %s 中找到预期的 t_* 和 x_* 变量。', data_filename);
end

t_var_name = var_names{t_var_index};
x_var_name = var_names{x_var_index};

fprintf('从文件 %s 中读取变量: %s 和 %s\n', data_filename, t_var_name, x_var_name);

% 从结构体中获取数据
t_full = loaded_data.(t_var_name); % <-- 修改：使用结构体字段访问
x_full = loaded_data.(x_var_name); % <-- 修改：使用结构体字段访问

% 检查加载的数据是否正确
if ~isnumeric(t_full) || ~isnumeric(x_full) || isempty(t_full) || isempty(x_full)
     error('从 %s 加载的变量 %s 或 %s 不是有效的数值数组。', data_filename, t_var_name, x_var_name);
end
if size(t_full,1) ~= size(x_full,1)
     error('时间向量 (%s) 和状态矩阵 (%s) 的行数不匹配。', t_var_name, x_var_name);
end

% 确定 dt (如果文件中有 dt 字段，优先使用；否则计算)
if isfield(loaded_data, 'dt') && isnumeric(loaded_data.dt) && isscalar(loaded_data.dt)
    dt = loaded_data.dt;
    fprintf('使用文件中存储的 dt = %.4f\n', dt);
else
    dt = t_full(2) - t_full(1); % 尝试从时间向量计算dt
    fprintf('从时间向量计算得到 dt = %.4f\n', dt);
    % 注意：确保 dt 被传递或保存给后续步骤，这里需要保存到临时文件
end


% 分割训练/测试数据 (从末尾取测试数据)
num_total_points = length(t_full);
num_test_points = floor(test_data_ratio * num_total_points);
num_train_val_points = num_total_points - num_test_points;

% --- >> 在这里添加以下代码 << ---
% 执行实际的数据分割
t_train_val = t_full(1:num_train_val_points);
x_train_val = x_full(1:num_train_val_points, :);
t_test = t_full(num_train_val_points+1:end);
x_test = x_full(num_train_val_points+1:end, :);
% --- >> 添加结束 << ---

% 创建一个临时的训练数据文件 (只包含训练/验证部分)
train_val_data_filename = sprintf('%s_train_val_temp.mat', system_choice);
% --- 确保 dt 被保存到临时文件 ---
% 现在 t_train_val 和 x_train_val 应该存在了
save(train_val_data_filename, 't_train_val', 'x_train_val', 'dt'); % <-- Line 111
fprintf('已将数据分割为 %d 个训练/验证点 和 %d 个测试点。\n', num_train_val_points, num_test_points);
% --- 4. 配置训练选项 ---
execution_environment = 'cpu';
if use_gpu && (gpuDeviceCount > 0) % 检查是否有可用GPU
    execution_environment = 'gpu';
    fprintf('检测到 GPU，将使用 GPU 进行训练。\n');
else
    fprintf('将使用 CPU 进行训练。\n');
end

train_options = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'InitialLearnRate', initial_learn_rate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', learn_rate_drop_period, ...
    'LearnRateDropFactor', learn_rate_drop_factor, ...
    'GradientThreshold', gradient_threshold, ...
    'Shuffle', 'every-epoch', ... % 在每个 epoch 开始时打乱训练数据
    'Plots', 'training-progress', ... % 显示训练进度图
    'Verbose', true, ...          % 在控制台显示训练信息
    'ExecutionEnvironment', execution_environment);
    % ValidationData 和 ValidationFrequency 会在 koopman_dnn_train 中根据 val_split 设置

% --- 5. 训练 Koopman DNN ---
model_filename = sprintf('dnn_model_%s.mat', system_choice);
[net, train_info, norm_params] = koopman_dnn_train(train_val_data_filename, system_choice, hidden_units, train_options, validation_split);

% 删除临时训练文件
delete(train_val_data_filename);

% --- 6. 运行 Koopman 观测器 ---
state_dim = size(x_test, 2);
if use_true_x0_for_observer
    x_hat0 = x_test(1, :); % 使用测试集的第一个真实状态作为初始估计
    fprintf('使用测试集的真实初始状态作为观测器起点。\n');
else
    x_hat0 = zeros(1, state_dim) + observer_noise_level * randn(1, state_dim); % 零向量加噪声
    fprintf('使用带噪声的零向量作为观测器初始起点。\n');
end

% 如果 L 被指定，则使用校正；否则纯预测
if isempty(L)
    fprintf('运行纯预测观测器 (L = [])...\n');
else
    fprintf('运行带校正项的观测器 (L provided)...\n');
    % 确保 L 的维度正确 (D x M)
     num_measurements = size(C, 1);
     if ~isequal(size(L), [state_dim, num_measurements])
         warning('提供的 L 矩阵维度不正确! 将使用纯预测。');
         L = []; % 重置为纯预测
     end
end

[t_hat, x_hat, y_test] = koopman_observer(model_filename, t_test, x_test, C, x_hat0, L);

% --- 7. 评估观测器性能 ---
[rmse_overall, rmse_per_state] = evaluate_observer(t_test, x_test, t_hat, x_hat, system_choice, C);

fprintf('\n--- 实验完成 for %s ---\n', system_choice);
fprintf('总体 RMSE: %.4f\n', rmse_overall);
disp('各状态 RMSE:');
disp(rmse_per_state);
fprintf('--------------------------\n');

% --- 9. (可选) 清理路径 ---
% rmpath('utils');