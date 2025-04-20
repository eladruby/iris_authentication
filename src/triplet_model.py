import tensorflow as tf
from tensorflow.keras import Model

class TripletLoss:
    def __init__(self):
        self.inp = tf.keras.Input(shape=(None, None, 3))
        self.out = []

    def embedding(self):
        vgg16_fe = tf.keras.applications.VGG16(
            include_top=False, weights=None, input_tensor=self.inp
        )
        vgg16_fe.trainable = False
        out_conv = tf.keras.layers.Conv2D(
            filters=16, kernel_size=1, strides=1, padding='same',
            activation='relu', name='out_conv')(vgg16_fe.layers[-1].output)
        self.out = tf.keras.layers.GlobalMaxPooling2D()(out_conv)
        return Model(inputs=vgg16_fe.inputs, outputs=self.out, name='embedding')

    def fine_tune(self, model):
        model.trainable = True
        freeze_until = 14
        for layer in model.layers[:freeze_until + 1]:
            layer.trainable = False
        return Model(inputs=model.inputs, outputs=model.outputs)
