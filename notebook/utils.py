import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def evaluate_classifier(cls, test_features, test_labels):
    """评估分类器的正确率"""
    acc = sum(cls.predict(test_features) == test_labels) / len(test_labels)
    return acc

def evaluate_regressor(reg, x_test, y_test):
    """求回归模型预测结果与实际结果的均方误差"""
    return mean_squared_error(y_test, reg.predict(x_test))

def load_dataset(filename, seperator='\t', label_index=-1, concat=False):
    fr = open(filename)
    dataArr = []; labelArr = []
    for line in fr.readlines():
        lineArr = line.strip().split(seperator)
        dataArr.append([float(i) for i in lineArr[:label_index]])
        labelArr.append(float(lineArr[label_index]))
    dataArr = np.array(dataArr)
    labelArr = np.array(labelArr)
    fr.close()
    if not concat: return dataArr, labelArr
    else: return np.concatenate([dataArr, labelArr[:, None]], axis=1)

def plot_scatter(data, labels=None):
    """将两列数据绘制成散点图
    labels: str | list(str) | None
    """
    if len(np.shape(labels)) == 0:
        x = data[:, 0]; y = data[:, 1]
        plt.scatter(x, y, label=labels)
    else:
        unique_ordered_labels = sorted(list(set(labels)))
        num_labels = len(unique_ordered_labels)
        marker_m = ['o', 'x', '.', '*']
        color_m = ['r', 'g', 'b']
        for i in range(num_labels):
            label = unique_ordered_labels[i]
            x = data[labels == label, 0]
            y = data[labels == label, 1]
            plt.scatter(x, y, marker=marker_m[i % len(marker_m)], c=color_m[i % len(color_m)], label=f'label={label}')
    
    return plt

def generate_regression_data(min=0, max=10, sample_num=50, f=lambda x: 2*x + 3):
    """生成regression算法需要使用到的数据样本"""
    x = np.linspace(min, max, sample_num)
    y = f(x) + np.random.normal(0, 1, size=sample_num)
    return x, y

def split_data_arr(data_arr, test_size=0.1, shuffle=True):
    """划分数据集"""
    x_train, x_test, y_train, y_test = train_test_split(data_arr[:, 0].reshape(-1, 1), data_arr[:, 1], test_size=test_size, shuffle=shuffle)
    return x_train, x_test, y_train, y_test
