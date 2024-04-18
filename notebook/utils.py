import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import keras
from keras import backend as K
from keras import datasets, layers, optimizers, utils, \
        models, losses, metrics, regularizers, applications

import random
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_images(images, rows, cols, cmap='gray'):
    """展示训练图片"""
    assert images.shape[0] == rows * cols, 'rows or cols wrong!'
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j + 1
            plt.subplot(rows, cols, index)
            plt.imshow(images[index - 1], cmap=cmap)

def plot_history_dict(history_dict, all_labels=[['loss', 'val_loss'], ['accuracy', 'val_accuracy']]):
    """将训练过程中的数据绘制成折线图"""
    # plot loss figure
    # 循环遍历所有的键和值，为每个键绘制一条线
    n_fig = len(all_labels)
    i = 1
    for labels in all_labels:
        plt.subplot(n_fig, 1, i)
        for label in labels:
            # 随机生成线条颜色和样式
            color = (random.random(), random.random(), random.random())
            linewidth = random.uniform(0.5, 2)
            linestyle = random.choice(['-', '--', '-.', ':'])
            # 绘制线条
            plt.plot(history_dict[label], label=label, color=color, linewidth=linewidth, linestyle=linestyle)
            plt.xlabel('loss' if 'loss' in label else 'accuracy')
        i += 1
        plt.legend()
    plt.show()

def vectorize_data(data, dimension=10000):
    """将输入的文本数据编码成one-hot vector数组"""
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(data), dimension))
    for i, sequence in enumerate(data):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

def train_net(net, 
              epochs: int, 
              batch_size: int, 
              x_train, 
              y_train, 
              x_test=None, 
              y_test=None, 
              val_ratio=None):
    """模型训练"""
    if val_ratio is None:
        history = net.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
        )
    else:
        num_total = len(x_train)
        n_train = int(num_total * (1 - val_ratio))
        part_x_train, x_val = x_train[:n_train], x_train[n_train:]
        part_y_train, y_val = y_train[:n_train], y_train[n_train:]
        history = net.fit(
            part_x_train,
            part_y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val)
        )
    acc = None
    if x_test is not None and y_test is not None:
        acc = net.evaluate(x_test, y_test)
    
    return history, acc

def pad_sequences(seqs, maxlen):
    """padding seqenences to exact length"""
    num_seq = len(seqs)
    new_seqs = np.zeros((num_seq, maxlen))
    for i in range(num_seq):
        seq = seqs[i]
        if len(seq) < maxlen:
            new_seqs[i] = np.append(seq, [0] * (maxlen - len(seq)))
        else:
            new_seqs[i] = seq[:maxlen]
    return new_seqs