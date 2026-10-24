"""Small baseline and ImageNet-pretrained MobileNetV2."""

import tensorflow as tf

from task2.common import IMAGE_SIZE


def compile_binary(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
        jit_compile=False,
    )
    return model


def baseline():
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1 / 255.0, name="scale_0_1")(inputs)
    for number, filters in enumerate([16, 32, 64], start=1):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                                   name=f"conv_{number}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{number}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{number}")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"pool_{number}")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return compile_binary(tf.keras.Model(inputs, outputs, name="baseline_cnn"), 1e-3)


def transfer(dropout=0.3, weights="imagenet"):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False,
        weights=weights,
    )
    base.trainable = False
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    # Equivalent to mobilenet_v2.preprocess_input: [0,255] -> [-1,1].
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocess")(inputs)
    # Keep BatchNorm moving statistics fixed, including during fine-tuning.
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return compile_binary(tf.keras.Model(inputs, outputs, name="mobilenet_classifier"), 1e-3)


def backbone(model):
    return next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))


def unfreeze_last_block(model):
    base = backbone(model)
    base.trainable = True
    in_last_block = False
    for layer in base.layers:
        if layer.name.startswith("block_16_"):
            in_last_block = True
        layer.trainable = in_last_block and not isinstance(
            layer, tf.keras.layers.BatchNormalization,
        )
    compile_binary(model, 1e-5)


def unfreeze_last_layers(model, count, learning_rate):
    base = backbone(model)
    base.trainable = True
    cutoff = len(base.layers) - count
    for index, layer in enumerate(base.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization,
        )
    compile_binary(model, learning_rate)

