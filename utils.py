import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Activation, BatchNormalization, Dropout, Conv2D, Dense, MaxPool2D, Input, Flatten
from keras.regularizers import l2
import keras.backend as K
import itertools
from keras.preprocessing.image import ImageDataGenerator

def preprocess_imagenet_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}
    if x.shape==4:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            x = x[:, ::-1, :, :]
            # Zero-center by mean pixel
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
        else:
            # 'RGB'->'BGR'
            x = x[:, :, :, ::-1]
            # Zero-center by mean pixel
            x[:, :, :, 0] -= 103.939
            x[:, :, :, 1] -= 116.779
            x[:, :, :, 2] -= 123.68
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            x = x[::-1, :, :]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            # 'RGB'->'BGR'
            x = x[:, :, ::-1]
            # Zero-center by mean pixel
            x[:, :, 0] -= 103.939
            x[:, :, 1] -= 116.779
            x[:, :, 2] -= 123.68
    return x

def undo_preprocess_imagenet_input(x, data_format=None):

    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}
    if x.shape==4:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            # Zero-center by mean pixel
            x[:, 0, :, :] += 103.939
            x[:, 1, :, :] += 116.779
            x[:, 2, :, :] += 123.68
            x = x[:, ::-1, :, :]
        else:
            # 'RGB'->'BGR'
            # Zero-center by mean pixel
            x[:, :, :, 0] += 103.939
            x[:, :, :, 1] += 116.779
            x[:, :, :, 2] += 123.68
            x = x[:, :, :, ::-1]

    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            # Zero-center by mean pixel
            x[0, :, :] += 103.939
            x[1, :, :] += 116.779
            x[2, :, :] += 123.68
            x = x[::-1, :, :]

        else:
            # 'RGB'->'BGR'
            # Zero-center by mean pixel
            x[:, :, 0] += 103.939
            x[:, :, 1] += 116.779
            x[:, :, 2] += 123.68
            x = x[:, :, ::-1]

    return x

def directory_train_test_split(path, prop=0.2, sample=False):    
    subdirs = os.listdir(PATH)
    print subdirs
    if sample:
        val_path = os.path.join(os.path.split(PATH)[0], "sample")
    else:
        val_path = os.path.join(os.path.split(PATH)[0], "validation")
    if not os.path.exists(val_path):
        os.mkdir(val_path)
        for subdir in subdirs:
            os.mkdir(os.path.join(val_path,subdir))
        
    for subdir in subdirs:
        sub_path = os.path.join(PATH,subdir)
        files = os.listdir(sub_path)
        valid_size = int(prop*len(files))
        valid_sample = np.random.choice(files, valid_size, replace=False)
        if not sample:
            for v in valid_sample:
                os.rename(os.path.join(sub_path,v),os.path.join(os.path.join(val_path,os.path.split(sub_path)[-1]),v))
        else:
            for v in valid_sample:
                shutil.copy(os.path.join(sub_path,v),os.path.join(os.path.join(val_path,os.path.split(sub_path)[-1]),v))  

def relu(x): return Activation('relu')(x)

def dropout(x, p): return Dropout(p)(x) 

def bn(x): return BatchNormalization(axis=-1)(x)

def relu_bn(x): return relu(bn(x))

def conv_layer(x, filters, k_size=(3,3), l2_penalty=0.0, drop_rate=0.0):
    x = Conv2D(filters=filters, kernel_size=k_size, padding="same", kernel_regularizer=l2(l2_penalty))(x)
    return dropout(x, drop_rate)

def onehot(x):
    return to_categorical(x)

def onehot_to_integer(x):
    integers = []
    for row in x:
        integers.append(np.argmax(row))
    return np.array(integers).reshape(-1,1)

def normalize_img(X):
    return X/255.

def standardize_imgs(X, params = None, return_params=True):
    assert X.shape[-1] == 3
    if not params:
        mu = np.mean(X, axis=(0,1,2))
        std = np.std(X, axis=(0,1,2))
    else:
        mu, std = params
    for c in range(3):
        X[:,:,:,c] = (X[:,:,:,c]-mu[c])/std[c]
    if return_params: return mu, std

def un_standardize(a, params):
    mu, std = params
    trans = []
    for c in range(3):
        trans.append((a[:,:,c]*std[c])+mu[c])
    return np.stack(trans, axis=-1)
    
def plot_img_array(a, params=None):
    if not params:
        plt.imshow(a)
    else:
        plt.imshow(un_standardize(a, params=params))
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]

def image_data_aug(rot=0., width_shift=0., height_shift=0., shear=0., zoom=0., h_flip=False, v_flip=False, fill="constant", cval=0.):
    return ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=rot,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        shear_range=shear,
        zoom_range=zoom,
        channel_shift_range=0.,
        fill_mode=fill,
        cval=cval,
        horizontal_flip=h_flip,
        vertical_flip=v_flip,
        rescale=None,
        preprocessing_function=None)