def vgg16_improved(input_shape=(112, 112, 5), lr=1e-5):
    model = Sequential([
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        MaxPool2D((2, 2), strides=(2, 2)),

        Conv2D(64, (3, 3), padding="same", activation="relu"),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        MaxPool2D((2, 2), strides=(2, 2)),

        Conv2D(128, (3, 3), padding="same", activation="relu"),
        Conv2D(128, (3, 3), padding="same", activation="relu"),
        MaxPool2D((2, 2), strides=(2, 2)),

        Conv2D(256, (3, 3), padding="same", activation="relu"),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        MaxPool2D((2, 2), strides=(2, 2)),

        Conv2D(256, (3, 3), padding="same", activation="relu"),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        MaxPool2D((2, 2), strides=(2, 2)),

        GlobalAveragePooling2D(),

        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dense(1, activation='linear'),
    ])

    sgd = optimizers.SGD(
        learning_rate=lr,
        decay=1e-4,
        momentum=0.9,
        nesterov=True
    )
    model.compile(optimizer=sgd, loss="mean_absolute_percentage_error")
    return model