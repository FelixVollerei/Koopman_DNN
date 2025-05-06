% evaluate_observer.m
function [rmse_overall, rmse_per_state] = evaluate_observer(t_true, x_true, t_hat, x_hat, system_name, C)
    % 评估 Koopman 观测器的性能
    % 输入:
    %   t_true, x_true: 真实时间和状态 (N x D)
    %   t_hat, x_hat: 估计时间和状态 (N x D)
    %   system_name: 系统名称 (用于绘图)
    %   C: (可选) 测量矩阵 (M x D), 用于区分可测/不可测状态
    % 输出:
    %   rmse_overall: 总体 RMSE
    %   rmse_per_state: 每个状态的 RMSE (1 x D)
    %   (生成图表)

    fprintf('开始评估观测器性能 for %s...\n', system_name);

    % --- 1. 检查数据维度和时间对齐 ---
    if ~isequal(size(x_true), size(x_hat))
        error('真实状态和估计状态的维度不匹配!');
    end
    % 可选: 如果 t_true 和 t_hat 可能不同，需要插值对齐
    if ~isequal(t_true, t_hat)
        fprintf('警告: 真实时间和估计时间向量不完全一致。尝试使用插值...\n');
        x_hat = interp1(t_hat, x_hat, t_true, 'linear', 'extrap');
        t_hat = t_true; % 以真实时间为基准
    end

    [N, state_dim] = size(x_true);

    % --- 2. 计算误差 ---
    error = x_true - x_hat;

    % --- 3. 计算 RMSE ---
    rmse_per_state = rms(error, 1); % 按列计算 RMSE (每个状态)
    rmse_overall = rms(error(:));   % 计算所有误差的总体 RMSE

    fprintf('观测器性能评估结果:\n');
    fprintf('  - 总体 RMSE: %.4f\n', rmse_overall);
    for i = 1:state_dim
        fprintf('  - 状态 %d RMSE: %.4f\n', i, rmse_per_state(i));
    end

    % --- 4. 可视化结果 ---
    fprintf('生成可视化图表...\n');
    plot_results(t_true, x_true, x_hat, error, rmse_per_state, system_name, C);

    fprintf('评估完成。\n');
end


% --- 辅助绘图函数 (可以放在 utils/plot_results.m 或这里) ---
function plot_results(t, x_true, x_hat, error, rmse_per_state, system_name, C)
    [N, state_dim] = size(x_true);
    figure('Name', ['Observer Performance: ', system_name], 'Position', [100, 100, 1200, 800]);

    % 区分可测和不可测状态
    measured_states = [];
    unmeasured_states = 1:state_dim;
    if nargin == 7 && ~isempty(C)
       measured_states = find(sum(abs(C), 1) > 1e-6); % 找到 C 中非零列对应的状态索引
       unmeasured_states(measured_states) = [];
    end

    % 绘制状态轨迹对比
    subplot(2, 2, 1);
    hold on;
    colors = lines(state_dim);
    h_true = []; h_hat = [];
    for i = 1:state_dim
        h_true(i) = plot(t, x_true(:, i), '-', 'Color', colors(i,:), 'LineWidth', 1.5);
        h_hat(i) = plot(t, x_hat(:, i), '--', 'Color', colors(i,:)*0.7, 'LineWidth', 1.5); % 虚线表示估计
    end
    hold off;
    xlabel('Time (s)');
    ylabel('State Value');
    title(['True (solid) vs Estimated (dashed) States']);
    legend_labels = {};
    for i = 1:state_dim
       legend_labels{end+1} = sprintf('x_{%d} True', i);
       legend_labels{end+1} = sprintf('x_{%d} Est', i);
    end
    % legend(legend_labels, 'Location', 'best'); % 可能太多了，只显示前几个或不显示
    grid on;

    % 绘制误差随时间变化
    subplot(2, 2, 3);
    hold on;
    for i = 1:state_dim
        plot(t, error(:, i), 'Color', colors(i,:));
    end
    hold off;
    xlabel('Time (s)');
    ylabel('Estimation Error');
    title('Estimation Error (True - Estimated)');
    legend(arrayfun(@(i) sprintf('Error x_{%d}', i), 1:state_dim, 'UniformOutput', false), 'Location', 'best');
    grid on;

    % 绘制二维/三维相图对比
    subplot(2, 2, 2);
    if state_dim >= 3
        plot3(x_true(:,1), x_true(:,2), x_true(:,3), 'b-', 'LineWidth', 1.5);
        hold on;
        plot3(x_hat(:,1), x_hat(:,2), x_hat(:,3), 'r--', 'LineWidth', 1.5);
        hold off;
        xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
        title('Phase Portrait (True: blue, Estimated: red)');
        legend('True Trajectory', 'Estimated Trajectory');
    elseif state_dim == 2
        plot(x_true(:,1), x_true(:,2), 'b-', 'LineWidth', 1.5);
        hold on;
        plot(x_hat(:,1), x_hat(:,2), 'r--', 'LineWidth', 1.5);
        hold off;
        xlabel('x_1'); ylabel('x_2');
        title('Phase Portrait (True: blue, Estimated: red)');
        legend('True Trajectory', 'Estimated Trajectory');
    else
        plot(t, x_true(:,1), 'b-'); hold on; plot(t, x_hat(:,1), 'r--'); hold off;
        title('State 1 Trajectory'); xlabel('Time'); ylabel('x_1');
    end
    grid on;
    axis equal;

    % 绘制 RMSE 条形图
    subplot(2, 2, 4);
    bar_colors = zeros(state_dim, 3);
    bar_labels = cell(1, state_dim);
     for i = 1:state_dim
         if ismember(i, measured_states)
             bar_colors(i,:) = [0 0.8 0]; % 绿色表示可测
             bar_labels{i} = sprintf('x_{%d} (Meas.)', i);
         else
             bar_colors(i,:) = [0.8 0 0]; % 红色表示不可测
             bar_labels{i} = sprintf('x_{%d} (Unmeas.)', i);
         end
     end
    b = bar(rmse_per_state);
    % 自定义颜色 (需要 R2019b 或更高版本)
    try
      b.FaceColor = 'flat';
      b.CData = bar_colors;
    catch
      disp('Bar coloring requires MATLAB R2019b or later.')
    end
    set(gca, 'XTickLabel', bar_labels);
    ylabel('RMSE');
    title('RMSE per State');
    grid on;

    sgtitle(['Koopman DNN Observer Evaluation for ', system_name], 'FontSize', 14, 'FontWeight', 'bold'); % Super title
end