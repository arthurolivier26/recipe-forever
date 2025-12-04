"""
custom_layers.py — Custom layers and custom losses for the Transformer model.
Fully compatible with TensorFlow 2.x and Keras 3.x.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------
#                   SERIALIZATION COMPATIBILITY HELPERS
# ---------------------------------------------------------------------
# Ensures custom objects work regardless of the environment:
# - Keras 3 standalone
# - TensorFlow ≥ 2.16
# - TensorFlow ≤ 2.15
# - Fallback: dummy decorator (no-op), still loads for inference
try:
    # Keras 3 standalone API
    from keras import saving
    register_serializable = saving.register_keras_serializable
except (ImportError, AttributeError):
    try:
        # TensorFlow 2.16+ uses tf.keras.saving
        from tensorflow.keras import saving
        register_serializable = saving.register_keras_serializable
    except (ImportError, AttributeError):
        try:
            # TF 2.15 and older
            register_serializable = tf.keras.utils.register_keras_serializable
        except AttributeError:
            # Last backup: no-op decorator
            def register_serializable(package=None, name=None):
                def decorator(cls):
                    return cls
                return decorator


# ============================================================
#                    CUSTOM LAYER — POSITIONAL EMBEDDING
# ============================================================

@register_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    """
    Learnable positional embedding layer for Transformer architectures.

    The layer learns one positional embedding per time step (0 → max_len-1)
    and adds it to the input embeddings to inject sequence order.

    Args:
        max_len (int): Maximum sequence length.
        d_model (int): Dimension of the model / embedding size.
    """

    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        """
        Add positional encodings to the input sequence.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor of same shape as x, enriched with positional information.
        """
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        learned_positions = self.pos_emb(positions)  # (max_len, d_model)
        return x + tf.expand_dims(learned_positions, 0)

    def get_config(self):
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
        })
        return config


# ============================================================
#                        CUSTOM LOSSES
# ============================================================

@register_serializable(package="Custom")
def masked_cosine_loss(y_true, y_pred):
    """
    Cosine similarity loss with masking to ignore padding positions.

    Used for recipe embeddings (384D) where the goal is to maximize
    cosine similarity between prediction and target vectors.

    Args:
        y_true: (batch, seq_len, embed_dim)
        y_pred: (batch, seq_len, embed_dim)

    Returns:
        Scalar loss value averaged over valid (non-padding) positions.
    """
    y_true_norm = tf.math.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.math.l2_normalize(y_pred, axis=-1)

    similarity = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)

    # Mask padding (where y_true is zero)
    valid_mask = tf.cast(tf.reduce_sum(tf.abs(y_true), axis=-1) > 1e-4, tf.float32)

    loss = (1.0 - similarity) * valid_mask

    return tf.reduce_sum(loss) / (tf.reduce_sum(valid_mask) + 1e-6)


@register_serializable(package="Custom")
def masked_huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber loss with masking for nutritional predictions.

    Huber loss is quadratic for small errors and linear for large ones,
    making it more robust than MSE to outliers.

    Args:
        y_true: (batch, seq_len, 2) — [protein, calories]
        y_pred: Same shape as y_true
        delta: Threshold at which loss transitions from quadratic to linear

    Returns:
        Scalar masked Huber loss.
    """
    error = y_true - y_pred
    is_small = tf.abs(error) <= delta

    sq_loss = 0.5 * tf.square(error)
    lin_loss = delta * (tf.abs(error) - 0.5 * delta)

    raw_loss = tf.reduce_mean(tf.where(is_small, sq_loss, lin_loss), axis=-1)

    valid_mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 1e-4, tf.float32)

    return tf.reduce_sum(raw_loss * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-6)


@register_serializable(package="Custom")
def masked_mse_loss(y_true, y_pred):
    """
    Standard Mean Squared Error for taste score prediction (0–1 range).

    Masking is intentionally *not* applied here: the model is trained to
    predict 0 for padding positions.

    Args:
        y_true: (batch, seq_len, 1)
        y_pred: Same shape

    Returns:
        Scalar MSE loss.
    """
    raw_loss = tf.square(y_true - y_pred)
    return tf.reduce_mean(raw_loss)


# ============================================================
#                MODEL LOADING WITH CUSTOM OBJECTS
# ============================================================

def load_transformer_with_custom_objects():
    """
    Load the pre-trained Transformer model while automatically
    registering all custom layers and loss functions.

    Returns:
        A Keras model ready for inference.
    """
    model = tf.keras.models.load_model(
        "models/TRANSFORMER_MODEL_PRETRAIN.keras",
        custom_objects=CUSTOM_OBJECTS,
        compile=False
    )
    return model


# ============================================================
#                 GLOBAL EXPORT OF CUSTOM OBJECTS
# ============================================================

CUSTOM_OBJECTS = {
    "PositionalEmbedding": PositionalEmbedding,
    "masked_cosine_loss": masked_cosine_loss,
    "masked_huber_loss": masked_huber_loss,
    "masked_mse_loss": masked_mse_loss,
}


def get_custom_objects():
    """
    Return a copy of the dictionary containing all custom Keras objects.
    Useful when loading models via keras.models.load_model().

    Returns:
        Dict[str, Any]: Custom layer/loss registry.
    """
    return CUSTOM_OBJECTS.copy()


# ============================================================
#                           SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing custom layers and losses...")
    print(f"TensorFlow version: {tf.__version__}")

    # Test PositionalEmbedding
    pos_emb = PositionalEmbedding(max_len=141, d_model=384)
    test_input = tf.random.normal((2, 141, 384))
    output = pos_emb(test_input)
    print(f"✅ PositionalEmbedding: input {test_input.shape} -> output {output.shape}")

    # Test masked_cosine_loss
    y_true = tf.random.normal((2, 10, 384))
    y_pred = tf.random.normal((2, 10, 384))
    loss = masked_cosine_loss(y_true, y_pred)
    print(f"✅ masked_cosine_loss: {loss.numpy():.4f}")

    # Test masked_huber_loss
    y_true = tf.random.normal((2, 10, 2))
    y_pred = tf.random.normal((2, 10, 2))
    loss = masked_huber_loss(y_true, y_pred)
    print(f"✅ masked_huber_loss: {loss.numpy():.4f}")

    # Test masked_mse_loss
    y_true = tf.random.uniform((2, 10, 1), 0, 1)
    y_pred = tf.random.uniform((2, 10, 1), 0, 1)
    loss = masked_mse_loss(y_true, y_pred)
    print(f"✅ masked_mse_loss: {loss.numpy():.4f}")

    print("\nAll tests passed! ✅")
