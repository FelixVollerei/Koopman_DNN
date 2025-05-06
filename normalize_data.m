% utils/normalize_data.m
function [normalized_data, norm_params] = normalize_data(data)
    % 对数据进行 Z-score 归一化 (按列处理)
    % 输入: data (N x D) - N个样本, D个特征
    % 输出:
    %   normalized_data (N x D) - 归一化后的数据
    %   norm_params - 包含均值(mu)和标准差(sigma)的结构体

    mu = mean(data, 1);
    sigma = std(data, 0, 1);

    % 处理标准差为零的情况（特征值恒定）
    sigma(sigma == 0) = 1;

    normalized_data = (data - mu) ./ sigma;

    norm_params.mu = mu;
    norm_params.sigma = sigma;
end