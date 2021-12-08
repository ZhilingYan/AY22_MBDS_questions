import numpy as np
import struct
import os
def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def fiter_mnist(train_images, train_labels, target_num=[2,7]):
    feature_num = len(train_images)
    train_images_filter = []
    train_labels_filter = []
    for i in range(feature_num):
        image = np.reshape(train_images[i], [28, 28])

        image = image[np.newaxis, np.newaxis, :, :,] / 256
        label = train_labels[i]
        if label in target_num:
            train_images_filter.append(image)
            if label == 2:
                train_labels_filter.append(0)
            else:
                train_labels_filter.append(1)
    return train_images_filter, train_labels_filter


if __name__ == '__main__':
    train_path = 'train'
    train_images, train_labels = load_mnist_train(train_path, kind='train')
    train_images_filter, train_labels_filter = fiter_mnist(train_images, train_labels, taget_num=[2,7])
    test_path = 'test'
    test_images, test_labels = load_mnist_test(test_path, kind='t10k')
    test_images_filter, test_labels_filter = fiter_mnist(test_images, test_labels, taget_num=[2,7])
    print(train_labels_filter[:10])
    print(test_labels_filter[:10])
