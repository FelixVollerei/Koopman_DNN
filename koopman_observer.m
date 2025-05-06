% koopman_observer.m
function [t_hat, x_hat, y_test] = koopman_observer(model_file, t_test, x_test, C, x_hat0, L)
    % 使用训练好的 DNN 实现 Koopman 状态观测器
    % 输入:
    %   model_file: 包含 net, norm_params, dt 的模型文件
    %   t_test: 测试数据时间向量 (N_test x 1)
    %   x_test: 测试数据真实状态 (N_test x D)
    %   C: 测量矩阵 (M x D), y = C * x' (M 个测量值, D 个状态)
    %   x_hat0: 初始状态估计 (1 x D)
    %   L: (可选) 观测器增益 (D x M)。如果为空或未提供，则不进行校正。
    % 输出:
    %   t_hat: 估计的时间向量 (与 t_test 相同)
    %   x_hat: 估计的状态轨迹 (N_test x D)
    %   y_test: 真实的测量值 (N_test x M)

    fprintf('运行 Koopman DNN 观测器...\n');

    % --- 1. 加载模型和参数 ---
    fprintf('加载模型: %s\n', model_file);
    load(model_file, 'net', 'norm_params', 'dt'); % 确保 dt 也被加载
    state_dim = net.Layers(1).InputSize;
    num_measurements = size(C, 1);

    % --- 2. 准备测试数据和初始化 ---
    [N_test, D_test] = size(x_test);
    if D_test ~= state_dim
        error('测试数据维度 (%d) 与模型输入维度 (%d) 不匹配!', D_test, state_dim);
    end
    if size(x_hat0, 2) ~= state_dim
       error('初始估计维度 (%d) 与模型输入维度 (%d) 不匹配!', size(x_hat0, 2), state_dim);
    end
     if size(C, 2) ~= state_dim
        error('测量矩阵 C 的列数 (%d) 与状态维度 (%d) 不匹配!', size(C, 2), state_dim);
    end

    t_hat = t_test;
    x_hat = zeros(N_test, state_dim);
    x_hat(1, :) = x_hat0;

    % 计算真实测量值 y = C * x' -> y' = x * C'
    y_test = x_test * C'; % (N_test x D) * (D x M) = (N_test x M)

    % 检查是否提供增益 L
    use_correction = (nargin == 6 && ~isempty(L));
    if use_correction
        fprintf('使用带校正项 (增益 L) 的观测器。\n');
         if ~isequal(size(L), [state_dim, num_measurements])
             error('观测器增益 L 的维度应为 (%d x %d)，当前为 (%d x %d)', ...
                   state_dim, num_measurements, size(L,1), size(L,2));
         end
    else
        fprintf('使用纯预测观测器 (无校正项 L)。\n');
        L = zeros(state_dim, num_measurements); % 确保 L 存在且为零
    end

    % --- 3. 运行观测器时间步进 ---
    fprintf('开始状态估计...\n');
    h_wait = waitbar(0, '运行观测器...');
    for k = 1:(N_test - 1)
        % a. 获取当前估计
        x_hat_k = x_hat(k, :); % (1 x D)

        % b. 归一化当前估计 (使用训练时的参数)
        x_hat_k_norm = (x_hat_k - norm_params.mu) ./ norm_params.sigma; % (1 x D)

        % c. 使用 DNN 预测下一步 (归一化空间)
        %    网络输入需要 D x 1
        x_hat_kplus1_pred_norm = predict(net, x_hat_k_norm); % (D x 1)
        %x_hat_kplus1_pred_norm = x_hat_kplus1_pred_norm'; % 转回 (1 x D)

        % d. 反归一化预测结果
        x_hat_kplus1_pred = denormalize_data(x_hat_kplus1_pred_norm, norm_params); % (1 x D)

        % e. (可选) 校正步骤
        if use_correction
            % 获取 k+1 时刻的真实测量值
            y_kplus1 = y_test(k+1, :); % (1 x M)
            % 计算 k+1 时刻的预测测量值 y_hat = C * x_hat'
            y_hat_kplus1_pred = x_hat_kplus1_pred * C'; % (1 x D) * (D x M) = (1 x M)
            % 计算测量误差
            measurement_error = y_kplus1 - y_hat_kplus1_pred; % (1 x M)
            % 更新状态估计: x_hat(k+1) = x_pred(k+1) + L * error'
            x_hat(k+1, :) = x_hat_kplus1_pred + (L * measurement_error')'; % (1 x D) + ((D x M) * (M x 1))'
        else
            % 无校正，直接使用预测值
            x_hat(k+1, :) = x_hat_kplus1_pred;
        end
        if mod(k, 100) == 0
             waitbar(k / (N_test-1), h_wait, sprintf('运行观测器... %d/%d', k, N_test-1));
        end
    end
    close(h_wait);
    fprintf('状态估计完成。\n');
end