% koopman_dnn_train.m
function [net, info, norm_params] = koopman_dnn_train(data_file, system_name, hidden_units, train_options, val_split)
    % 定义和训练用于逼近 Koopman 算子的 DNN    
    % 输入:   
    % data_file: .mat 文件路径 (包含 t, x, dt)    
    % system_name: 系统名称 (用于保存模型)   
    % hidden_units: 包含各隐藏层神经元数量的向量, e.g., [128, 64]    
    % train_options: trainingOptions 对象    
    % val_split: (可选) 验证集比例, e.g., 0.15    
    % 输出:    
    % net: 训练好的网络    
    % info: 训练信息    
    % norm_params: 归一化参数    
    % (保存模型文件 'dnn_model_[system_name].mat')    
    fprintf('开始训练 %s 系统的 Koopman DNN...\n', system_name);    
    % --- 1. 加载和准备数据 ---    
    fprintf('加载数据: %s\n', data_file);   
    loaded_data = load(data_file);    
    var_names = fieldnames(loaded_data);    
    t = loaded_data.(var_names{cellfun(@(s) contains(s,'t_'), var_names)}); % 找到时间变量    
    x = loaded_data.(var_names{cellfun(@(s) contains(s,'x_'), var_names)}); % 找到状态变量    
    if isfield(loaded_data, 'dt')    
        dt = loaded_data.dt;        
        fprintf('数据时间步长 dt = %.4f\n', dt);   
    else  
        dt = t(2)-t(1); % 尝试从时间向量计算dt       
        fprintf('从时间向量计算 dt = %.4f\n', dt);   
    end    
    [N, state_dim] = size(x);    
    fprintf('数据点数: %d, 状态维度: %d\n', N, state_dim);    
    % --- 2. 归一化数据 ---    
    fprintf('归一化数据...\n');    
    [x_norm, norm_params] = normalize_data(x);    
    fprintf('归一化完成. 均值: %s, 标准差: %s\n', mat2str(norm_params.mu,3), mat2str(norm_params.sigma,3));    
    % --- 3. 准备 DNN 输入/输出对 ---    
    fprintf('准备 DNN 输入/输出对...\n');    
    [X_dnn_norm, Y_dnn_norm] = prepare_dnn_data(x_norm); % 使用归一化数据    
    % --- 4. 数据分割 (训练集/验证集) ---    
    num_samples = size(X_dnn_norm, 1);    
    if nargin < 5 || isempty(val_split) || val_split <= 0    
        % 不分割验证集        
        X_train = X_dnn_norm;        
        Y_train = Y_dnn_norm;       
        train_options.ValidationData = {}; % 清除验证数据选项       
        fprintf('使用所有 %d 个样本进行训练。\n', num_samples);     
    else    
        % 分割验证集 (时间序列数据，顺序分割)        
        num_val = floor(val_split * num_samples);        
        num_train = num_samples - num_val;       
        X_train = X_dnn_norm(1:num_train, :);       
        Y_train = Y_dnn_norm(1:num_train, :);  
        X_train = X_train';
        Y_train = Y_train';
        X_val = X_dnn_norm(num_train+1:end, :);      
        Y_val = Y_dnn_norm(num_train+1:end, :);     
        X_val = X_val';
        Y_val = Y_val';
        fprintf('调试信息：分割后 X_train 的维度: %s\n', mat2str(size(X_train)));       
        fprintf('调试信息：分割后 Y_train 的维度: %s\n', mat2str(size(Y_train)));        
        if exist('X_val', 'var') % 如果存在验证集，也检查一下        
            fprintf('调试信息：分割后 X_val 的维度: %s\n', mat2str(size(X_val)));            
            fprintf('调试信息：分割后 Y_val 的维度: %s\n', mat2str(size(Y_val)));        
        end        
        % Deep Learning Toolbox 需要 'features x observations' 格式        
        % 将验证数据打包成 cell array {InputData, ResponseData}        
        train_options.ValidationData = {X_val', Y_val'};        
        train_options.ValidationFrequency = max(1, floor(num_train / train_options.MiniBatchSize / 5)); % 每 1/5 epoch 验证一次        
        fprintf('数据分割: %d 训练样本, %d 验证样本。\n', num_train, num_val);        
    end    
    % --- 5. 定义 DNN 结构 ---    
    fprintf('定义 DNN 结构...\n');    
    layers = [featureInputLayer(state_dim, 'Name', 'input', 'Normalization', 'none')]; % 输入层    
    for i = 1:length(hidden_units)   
        layers = [layers        
        fullyConnectedLayer(hidden_units(i), 'Name', ['fc' num2str(i)])        
        reluLayer('Name', ['relu' num2str(i)])]; % 使用 ReLU 激活函数，可换成 tanhLayer 等        
        % dropoutLayer(0.1, 'Name', ['dropout' num2str(i)]) % (可选) 添加 Dropout 防止过拟合        
    end    
    layers = [layers    
    fullyConnectedLayer(state_dim, 'Name', 'output') % 输出层，维度与状态一致    
    regressionLayer('Name', 'regression')]; % 回归层，用于计算 MSE 损失    
    % --- 6. 训练网络 ---    
    fprintf('开始训练网络 (这可能需要一些时间)...\n');    
    % Deep Learning Toolbox 需要 'features x observations' 格式   
    fprintf('调试信息：传递给 trainNetwork 的 X_train'' 维度: %s\n', mat2str(size(X_train')));   
    fprintf('调试信息：传递给 trainNetwork 的 Y_train'' 维度: %s\n', mat2str(size(Y_train')));    
    fprintf('调试信息：期望的 Y_train'' 维度应该是: [%d, %d]\n', state_dim, size(X_train, 1)); % state_dim x num_train   
    [net, info] = trainNetwork(X_train', Y_train', layers, train_options);    
    fprintf('训练完成. 最终 RMSE: %.4f\n', info.TrainingRMSE(end));
    if ~isempty(train_options.ValidationData)    
        fprintf('最终验证 RMSE: %.4f\n', info.ValidationRMSE(end));    
    end    
    % --- 7. 保存模型 ---    
    model_filename = sprintf('dnn_model_%s.mat', system_name);   
    fprintf('保存训练好的模型到 %s...\n', model_filename);    
    save(model_filename, 'net', 'norm_params', 'system_name', 'dt');    
    fprintf('模型保存完毕。\n');
end