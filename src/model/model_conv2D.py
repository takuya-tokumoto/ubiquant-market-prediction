def correlation(x, y, axis=-2):
    """Metric returning the Pearson correlation coefficient of two tensors over some axis, default -2."""
    x = tf.convert_to_tensor(x)
    y = math_ops.cast(y, x.dtype)
    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis=axis)
    ysum = tf.reduce_sum(y, axis=axis)
    xmean = xsum / n
    ymean = ysum / n
    ###    不偏分散にしたら？？   ###
    
    xvar = tf.reduce_sum( tf.math.squared_difference(x, xmean), axis=axis)
    yvar = tf.reduce_sum( tf.math.squared_difference(y, ymean), axis=axis)

    cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
    corr = cov / tf.sqrt(xvar * yvar)
    return tf.constant(1.0, dtype=x.dtype) - corr

def get_model():
    features_inputs = tf.keras.Input((300, ), dtype=tf.float16)
    
    ## Dense 1 ##
    feature_x = layers.Dense(256, activation='swish', kernel_initializer = 'he_normal')(features_inputs)
    feature_x = layers.Dropout(0.1)(feature_x)
#     ## Dense 2 ##
#     feature_x = layers.Dense(256, activation='swish')(feature_x)
#     feature_x = layers.Dropout(0.1)(feature_x)
    ## convolution 1 ##
    feature_x = layers.Reshape((-1,1))(feature_x)
    feature_x = layers.Conv1D(filters=16, kernel_size=4, strides=1, padding='same', kernel_initializer = 'he_normal')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU(0.5)(feature_x)
    ## convolution 2 ##
    feature_x = layers.Conv1D(filters=16, kernel_size=4, strides=4, padding='same', kernel_initializer = 'he_normal')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU(0.5)(feature_x)
    ## convolution 3 ##
    feature_x = layers.Conv1D(filters=64, kernel_size=4, strides=1, padding='same', kernel_initializer = 'he_normal')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU(0.5)(feature_x)

    ## convolution2D 1 ##
    feature_x = layers.Reshape((64,64,1))(feature_x)
    feature_x = layers.Conv2D(filters=32, kernel_size=4, strides=1, padding='same', kernel_initializer = 'he_normal')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU(0.5)(feature_x)
    ## convolution2D 2 ##
    feature_x = layers.Conv2D(filters=32, kernel_size=4, strides=4, padding='same', kernel_initializer = 'he_normal')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU(0.5)(feature_x)
    ## convolution2D 3 ##
    feature_x = layers.Conv2D(filters=32, kernel_size=4, strides=4, padding='same', kernel_initializer = 'he_normal')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU(0.5)(feature_x)

    ## flatten ##
    feature_x = layers.Flatten()(feature_x)
    ## Dense 3 ##
    x = layers.Dense(512, activation='swish', kernel_regularizer="l2", kernel_initializer = 'he_normal')(feature_x)
    ## Dense 4 ##
    x = layers.Dropout(0.1)(x)
    ## Dense 5 ##    
    x = layers.Dense(128, activation='swish', kernel_regularizer="l2", kernel_initializer = 'he_normal')(x)
    x = layers.Dropout(0.1)(x)
    ## Dense 6 ##
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2", kernel_initializer = 'he_normal')(x)
    x = layers.Dropout(0.1)(x)
    ## Dense 7 ##
    output = layers.Dense(1)(x)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[features_inputs], outputs=[output])
    
    learning_sch = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.001,
        decay_steps = 9700,
        decay_rate = 0.98
    )
    adam = tf.keras.optimizers.Adam(learning_rate = learning_sch)
    
    model.compile(optimizer = adam, loss='mse', metrics=['mse', "mae", "mape", rmse, correlation])
    return model