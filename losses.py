"""
Define our custom loss function.
"""
import numpy as np
from keras import backend as K
import tensorflow as tf
import dill
import matplotlib.pyplot as plt

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        print(tf.reduce_any(y_true == -1) == True)
        if tf.reduce_any(y_true == -1) == True:
            return 0
        else:
            # Clip the prediction value to prevent NaN's and Inf's
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

            # Calculate Cross Entropy
            cross_entropy = -y_true * K.log(y_pred)

            # Calculate Focal Loss
            loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
            print(loss)
            # Compute mean loss in mini_batch
            print(K.mean(K.sum(loss, axis=-1)))
            return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def plot_loss(input_dir):
    f = open(input_dir, "r")
    dict_epoch = {}
    for line in f:
        if "e-" in line:
            line = line.replace("e-", "*")
        epoch, batch, d_loss, acc, g_loss1, g_loss2, g_loss3 = line.split('-')
        if "*" in d_loss:
            d_loss, mul = d_loss.split("*")
            d_loss = float(d_loss) * 10**(-int(mul))
        if "*" in g_loss3:
            g_loss3, mul = g_loss3.split("*")
            g_loss3 = float(g_loss3) * 10**(-int(mul))
        if epoch not in dict_epoch:
            dict_epoch[epoch]= {"d": [], "g": []}
        dict_epoch[epoch]["d"].append(d_loss)
        dict_epoch[epoch]["g"].append(g_loss3)
    list_d_loss = []
    list_epoch = []
    list_g_loss = []
    for key, val in dict_epoch.items():
        d_val = [float(e) for e in val["d"]]
        g_val = [float(e) for e in val["g"]]
        key = int(key)
        list_d_loss.append(sum(d_val)/len(d_val))
        list_g_loss.append(sum(g_val)/len(g_val))
        list_epoch.append(key)
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(list_epoch, list_d_loss, color ="red")
    plt.plot(list_epoch, list_g_loss, color ="blue")
    plt.legend(["Discriminator Loss", "Generator Loss (Combined)"])
    plt.savefig('combined_loss_epoch.png', bbox_inches='tight')



if __name__ == '__main__':

    # Test serialization of nested functions
    # bin_inner = dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25)))
    # print(bin_inner)

    # cat_inner = dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=.25)))
    # print(cat_inner)
    plot_loss('./loss_wgan.txt')
