import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tcn import TCN # Make sure this library is installed: pip install keras-tcn

def create_tcn_regression_model(input_shape, num_outputs, nb_filters, kernel_size, dilations, nb_stacks,
                                padding='causal', use_skip_connections=True, return_sequences=False,
                                dropout_rate=0.0, activation='relu',
                                kernel_initializer='he_normal', use_batch_norm=True, use_layer_norm=False,
                                use_l2_reg=False, l2_factor=0.001,
                                opt='adam', lr=0.002, model_name='tcn_regression_model'):

    input_layer = Input(shape=input_shape, name="Input_Layer")

    tcn_layer = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                    dilations=dilations, padding=padding, use_skip_connections=use_skip_connections,
                    dropout_rate=dropout_rate, return_sequences=return_sequences, activation=activation,
                    kernel_initializer=kernel_initializer, use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm, name=model_name)(input_layer)

    output_regularizer = l2(l2_factor) if use_l2_reg else None
    print(f"Output L2 Reg: {'Enabled (factor={})'.format(l2_factor) if use_l2_reg else 'Disabled'}")

    dense_output = Dense(num_outputs, name="Dense_Output_Regressor",
                         kernel_regularizer=output_regularizer)(tcn_layer)

    output_layer = Activation('linear', name="Linear_Output")(dense_output)

    model = Model(input_layer, output_layer, name=f"{model_name}_Compiled")

    if opt.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    elif opt.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0)
    else:
        print(f"Optimizer '{opt}' not recognized, defaulting to Adam.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    loss_func = 'mean_squared_error'
    metrics = ['mae', 'mse']
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    print(f"Model compiled with Loss: {loss_func}, Metrics: {metrics}, Optimizer LR: {lr}")

    return model