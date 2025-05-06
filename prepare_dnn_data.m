% utils/prepare_dnn_data.m
function [X_dnn, Y_dnn] = prepare_dnn_data(x)
    % 将时间序列数据 x 准备成 DNN 输入(X)和输出(Y)对
    % X = x(t)
    % Y = x(t+dt)
    % 输入: x (N x D) - N个时间点, D个状态维度
    % 输出:
    %   X_dnn (N-1 x D) - 输入数据
    %   Y_dnn (N-1 x D) - 目标数据 (下一时刻状态)

    X_dnn = x(1:end-1, :);
    Y_dnn = x(2:end, :);
end