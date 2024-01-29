from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Lambda
from config import image_shape, num_classes
from keras import Model, Input


def costume_model():
    input_layer = Input(shape=(image_shape, image_shape, 3))

    # Add a normalization layer to scale pixel values between 0 and 1
    normalized_input = Lambda(lambda y: y / 255.0)(input_layer)
    x = Conv2D(128, (3, 3), activation='relu')(normalized_input)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    return Model(input_layer, output_layer, name='costume_model')
