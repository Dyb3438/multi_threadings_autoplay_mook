# 该程序使用 TensorFlow 对 CNN 进行实现
# 图像预处理部分使用 OpenCV

import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 防止报出 TensorFlow 中的警告
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

charts = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11,
          'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22,
          'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33,
          'y': 34, 'z': 35}


# 以下代码用于实现卷积网络
class cnn_net:
    """
        Input_data
          ↓
        Conv1  →  ·   → ·
        Max_pool           ↓
          ↓               ↓
        Conv2              ↓
        Max_pool           ↓
          ↓             Conv4
        Conv3            Max_pool
        Max_pool           ↓
          ↓               ↓
         Fully connected layer
      (Activation function: Relu)
                  ↓
            Dropout layer
                  ↓
         Fully connected layer
                  ↓
              Prediction

        Conv1: 4*4  input:1   output:64
        Conv2: 3*3  input:64  output:32
        Conv3: 3*3  input:32  output:16
        Conv4: 1*1  input:64  output:16 (残差引进)
    """

    def __init__(self, is_training=False):
        self.batch_size = 36

        self.x_data = tf.placeholder(tf.float32, [None, 50, 50, 1])
        self.y_target = tf.placeholder(tf.float32, [None, 36])
        self.model_output = self.conv_net(self.x_data, is_training)
        self.prediction = tf.nn.softmax(self.model_output)
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.save_path = './saver/model.ckpt'

    def weight_init(self, shape, name):
        return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

    def bias_init(self, shape, name):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))

    def conv2d(self, input_data, conv_w):
        return tf.nn.conv2d(input_data, conv_w, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool(self, input_data, size):
        return tf.nn.max_pool(input_data, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='VALID')

    def conv_net(self, input_data, is_training=False):
        with tf.name_scope('conv1'):
            w_conv1 = self.weight_init([4, 4, 1, 64], 'conv1_w')  # 卷积核大小是 3*3 输入是 1 通道,输出为 64 通道
            b_conv1 = self.bias_init([64], 'conv1_b')
            h_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(input_data, w_conv1), b_conv1))
            self.h_pool1 = self.max_pool(h_conv1, 2)

        with tf.name_scope('conv2'):
            w_conv2 = self.weight_init([3, 3, 64, 32], 'conv2_w')  # 卷积核大小是 5*5 输入是64,输出为 32
            b_conv2 = self.bias_init([32], 'conv2_b')
            h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, w_conv2) + b_conv2)
            self.h_pool2 = self.max_pool(h_conv2, 2)

        with tf.name_scope('conv1_res'):
            w_conv1_res = self.weight_init([1, 1, 64, 16], 'conv1_res_w')
            b_conv1_res = self.bias_init([16], 'conv1_res_b')
            h_conv1_res = tf.nn.relu(self.conv2d(self.h_pool1, w_conv1_res) + b_conv1_res)
            self.h_pool1_res = self.max_pool(h_conv1_res, 5)

        with tf.name_scope('conv3'):
            w_conv3 = self.weight_init([3, 3, 32, 16], 'conv3_w')  # 卷积核大小是 5*5 输入是32,输出为 16
            b_conv3 = self.bias_init([16], 'conv3_b')
            h_conv3 = tf.nn.relu(self.conv2d(self.h_pool2, w_conv3) + b_conv3)
            self.h_pool3 = self.max_pool(h_conv3, 2)

        with tf.name_scope('fc1'):
            w_fc1 = self.weight_init([4 * 4 * 32, 64], 'fc1_w')  # 三层卷积后得到的图像大小为 28 * 12
            b_fc1 = self.bias_init([64], 'fc1_b')
            self.h_fc1 = tf.nn.relu(
                tf.matmul(tf.reshape(tf.concat([self.h_pool3, self.h_pool1_res], axis=3), [-1, 4 * 4 * 32]),
                          w_fc1) + b_fc1)
            if is_training:
                self.h_fc1 = tf.nn.dropout(self.h_fc1, 0.5)

        with tf.name_scope('fc2'):
            w_fc2 = self.weight_init([64, 36], 'fc2_w')
            b_fc2 = self.bias_init([36], 'fc2_b')
            h_fc2 = tf.matmul(self.h_fc1, w_fc2) + b_fc2

        return h_fc2


def load_images():
    img_list = []
    path = './fenge_mark'
    for path, _, files in os.walk(path):
        for file in files:
            file_path = path + '/' + file
            image = cv2.resize(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), (50, 50), interpolation=cv2.INTER_CUBIC)

            name = file[-5]
            img_list.append([name, image])
    return img_list


def FenLei(img_list):
    fenlei = {}
    for img_li in img_list:
        if img_li[0] in fenlei:
            fenlei[img_li[0]].append(img_li[1])
        else:
            fenlei.update({img_li[0]: [img_li[1]]})
    return fenlei


def Chart2Int(Chart):
    return charts[Chart]


def Int2Chart(int):
    for i, d in enumerate(charts):
        if i == int:
            return d
    return


def Rotate(img, rotate):
    height, width = img.shape
    matRotate = cv2.getRotationMatrix2D((width // 2, height // 2), rotate, 1)
    dst = cv2.warpAffine(img, matRotate, (width, height), borderValue=(255, 255, 255))
    dst = np.array(dst)
    return dst


def train_data():
    image_list = load_images()
    fenlei = FenLei(image_list)
    cnn = cnn_net(is_training=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=cnn.y_target, logits=cnn.model_output))

    optimizer = tf.train.AdamOptimizer(1e-5).minimize(loss)

    init = tf.global_variables_initializer()
    cnn.session.run(init)

    if os.path.exists('./saver/checkpoint'):
        cnn.saver.restore(cnn.session, cnn.save_path)

    train_loss = []
    train_accuracy = []

    for i in range(1000):  # 训练 100 次
        # 随机抽取训练元素 // 随机旋转 -5 , 5
        _list = [[i, Rotate(j, random.randint(-5, 5))] for i in fenlei for j in random.sample(fenlei[i], 5)]
        cnn.batch_size = len(_list)

        np_train_datas = [i[1][:, :, np.newaxis] / 256 for i in _list]

        train_labels_data = np.array([Chart2Int(i[0]) for i in _list])

        train_labels = tf.one_hot(train_labels_data, 36, on_value=1.0, off_value=0.0)

        # 计算训练集准确率
        train_correct_prediction = tf.equal(tf.argmax(cnn.model_output, 1), tf.argmax(train_labels, 1))

        train_accuracy_1 = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

        [_loss, _, accuracy] = cnn.session.run([loss, optimizer, train_accuracy_1],
                                               feed_dict={cnn.x_data: np_train_datas,
                                                          cnn.y_target: train_labels.eval(
                                                              session=cnn.session)})
        predicts = cnn.session.run(cnn.prediction, feed_dict={cnn.x_data: np_train_datas})
        predict = [np.unravel_index(predict.argmax(), predict.shape)[0] for predict in predicts]
        print('预测结果:', [Int2Chart(i) for i in predict])
        print('正确结果:', [i[0] for i in _list])

        train_loss.append(_loss)
        train_accuracy.append(accuracy)

        print('第 %d 次迭代:' % i)
        print('loss: %0.10f' % _loss)
        print('perdiction: %0.2f' % accuracy)

        if i % 100 == 0:
            cnn.saver.save(cnn.session, cnn.save_path)
            print('保存数据!')

    cnn.saver.save(cnn.session, cnn.save_path)
    print('保存数据!')
    cnn.session.close()
    plt.title('train loss')
    plt.plot(range(0, 1000), train_loss, 'b-')
    plt.show()
    plt.title('accuracy')
    plt.plot(range(0, 1000), train_accuracy, 'k-')
    plt.show()


def test():
    img_list = []
    path = './fenge'
    cnn = cnn_net()
    for path, _, files in os.walk(path):
        for file in files:
            file_path = path + '/' + file
            image = cv2.resize(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), (50, 50), interpolation=cv2.INTER_CUBIC)
            img_list.append(image)

    np_train_datas = np.empty(shape=(cnn.batch_size, 50, 50, 1), dtype='float32')

    for i in range(cnn.batch_size):
        np_train_datas[i] = img_list[i][:, :, np.newaxis] / 256  # 归一化g

    cnn.saver.restore(cnn.session, cnn.save_path)
    predicts = cnn.session.run(cnn.prediction, feed_dict={cnn.x_data: np_train_datas})
    predict = [np.unravel_index(predict.argmax(), predict.shape)[0] for predict in predicts]
    print('测试集预测:', [Int2Chart(i) for i in predict])
    cnn.session.close()


def main():
    #train_data()
    test()


if __name__ == "__main__":
    main()
