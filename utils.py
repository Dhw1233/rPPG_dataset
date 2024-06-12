from scipy.signal import butter, lfilter
import torch
import torch.nn as nn
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def psnr(img, img_g):
    criterionMSE = nn.MSELoss()  # .to(device)
    mse = criterionMSE(img, img_g)

    psnr = 10 * torch.log10(torch.tensor(1) / mse)  # 20 *
    return psnr

def calculate_statistics(parameters):
    # 转换为 numpy 数组以进行统计计算
    parameters_np = parameters.detach().cpu().numpy()
    mean = np.mean(parameters_np)
    std = np.std(parameters_np)
    return mean, std

def prune_weights(model,threshold):
    for name, parameter in model.named_parameters():
        if 'weight' in name:
            # 计算该层的平均值和标准差
            # mean, std = calculate_statistics(parameter)
            # 设置阈值
            # 执行剪枝操作
            parameter.data = torch.where(parameter.abs() < threshold, torch.zeros_like(parameter), parameter)
    return model
