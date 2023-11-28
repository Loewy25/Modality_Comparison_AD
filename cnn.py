from tensorflow.keras.layers import Conv3D, Input, LeakyReLU, Add, GlobalAveragePooling3D, Dense, Dropout, SpatialDropout3D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa


def convolution_block(x, filters, kernel_size=(3,3,3), strides=(1,1,1)):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = LeakyReLU()(x)
    return x

def context_module(x, filters):
    # First convolution block
    x = convolution_block(x, filters)
    # Dropout layer
    x = SpatialDropout3D(0.3)(x) 
    # Second convolution block
    x = convolution_block(x, filters)
    return x

def create_cnn_model():
    input_img = Input(shape=(128, 128, 128, 1))
    x = convolution_block(input_img, 16, strides=(1,1,1))
    conv1_out = x

    # Context 1
    x = context_module(x, 16)
    x = Add()([x, conv1_out])
    x = convolution_block(x, 32, strides=(2,2,2))
    conv2_out = x

    # Context 2
    x = context_module(x, 32)
    x = Add()([x, conv2_out])
    x = convolution_block(x, 64, strides=(2,2,2))
    conv3_out = x

    # Context 3
    x = context_module(x, 64)
    x = Add()([x, conv3_out])
    x = convolution_block(x, 128, strides=(2,2,2))
    conv4_out = x

    # Context 4
    x = context_module(x, 128)
    x = Add()([x, conv4_out])
    x = convolution_block(x, 256, strides=(2,2,2))
    
    # Context 5
    x = context_module(x, 256)

    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)

    # Dropout layer as described in the paper
    x = SpatialDropout3D(0.3)(x)   # The paper mentioned a dropout layer after GAP

    # Dense layer with 7 output nodes as described in the paper
    output = Dense(2, activation='softmax')(x) 

    model = Model(inputs=input_img, outputs=output)
    model.summary()

    return model
