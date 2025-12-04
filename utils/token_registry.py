"""
Token Registry – Management of special structural tokens for the Transformer.
Compatible with the REC_SYS_MAIN_PIPELINE notebook.
"""

import numpy as np
from typing import Dict, Optional


class TokenRegistry:
    """
    Registry of all special tokens used to structure meal plans and daily schedules.
    This is a clean and modular reimplementation of the class used in the notebook.
    """

    # Structure tokens
    START = "<START>"
    EOS = "<EOS>"

    DAY_START = "<DAY_START>"
    DAY_END = "<DAY_END>"

    # Meal tokens
    BREAKFAST = "<BREAKFAST>"
    LUNCH = "<LUNCH>"
    SNACK = "<SNACK>"
    DINNER = "<DINNER>"

    # Course-type tokens
    STARTER = "<STARTER>"
    MAIN_DISH = "<MAIN_DISH>"
    DESSERT = "<DESSERT>"

    def __init__(self, sentence_transformer=None):
        """
        Initialize the registry and optionally generate embeddings for all tokens.

        Args:
            sentence_transformer: Optional SentenceTransformer model used to
                                  generate vector embeddings for each token.
        """
        self.tokens = self._get_all_tokens()
        self._sentence_transformer = sentence_transformer

        if sentence_transformer is not None:
            self.embeddings = self._generate_embeddings()
            self.token_to_embedding = dict(zip(self.tokens, self.embeddings))
        else:
            self.embeddings = None
            self.token_to_embedding = {}

    def _get_all_tokens(self):
        """
        Retrieve all class attributes that represent tokens.
        A token is any UPPERCASE class-level string constant.
        """
        return [
            getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("_")
            and isinstance(getattr(self, attr), str)
            and attr.isupper()
        ]

    def _generate_embeddings(self):
        """
        Generate embeddings for all tokens using the SentenceTransformer.
        Returns:
            A NumPy array of embeddings, or None if no model is provided.
        """
        if self._sentence_transformer is None:
            return None
        return self._sentence_transformer.encode(self.tokens)

    def get_embedding(self, token: str) -> Optional[np.ndarray]:
        """
        Retrieve the embedding of a specific token.
        Returns None if embeddings are not loaded or the token is unknown.
        """
        return self.token_to_embedding.get(token)

    def get_tokens_dict(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary mapping token names without angle brackets
        to their embeddings.

        Example:
            "<BREAKFAST>" → "BREAKFAST"
        """
        return {
            token.replace("<", "").replace(">", ""): emb
            for token, emb in self.token_to_embedding.items()
        }

    def get_all_tokens_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Return a full token → embedding mapping (including brackets).
        """
        return self.token_to_embedding.copy()

    def set_embeddings_from_dict(self, embeddings_dict: Dict[str, np.ndarray]):
        """
        Load precomputed embeddings from an external dictionary.

        Args:
            embeddings_dict: Mapping {token_string: embedding_vector}
        """
        self.token_to_embedding = embeddings_dict.copy()
        self.tokens = list(embeddings_dict.keys())
        self.embeddings = np.array(list(embeddings_dict.values()))

    def __str__(self):
        return f"TokenRegistry with {len(self.tokens)} tokens"

    def __repr__(self):
        return f"TokenRegistry(tokens={self.tokens})"


# Friendly display mapping (for UI, logs, etc.)
TOKEN_DISPLAY = {
    "<START>": "🚀 Start",
    "<EOS>": "🛑 End",
    "<DAY_START>": "📅 Start of day",
    "<DAY_END>": "✅ End of day",
    "<BREAKFAST>": "☕ Breakfast",
    "<LUNCH>": "🌞 Lunch",
    "<SNACK>": "🍪 Snack",
    "<DINNER>": "🌙 Dinner",
    "<STARTER>": "🥗 Starter",
    "<MAIN_DISH>": "🥘 Main Course",
    "<DESSERT>": "🍰 Dessert"
}

# Token → meal type
TOKEN_TO_MEAL = {
    "<BREAKFAST>": "breakfast",
    "<LUNCH>": "lunch",
    "<SNACK>": "snack",
    "<DINNER>": "dinner"
}

# Token → course type
TOKEN_TO_COURSE = {
    "<STARTER>": "starter",
    "<MAIN_DISH>": "main",
    "<DESSERT>": "dessert"
}


def create_token_registry(sentence_transformer=None) -> TokenRegistry:
    """
    Factory function to create a TokenRegistry.

    Args:
        sentence_transformer: Optional SentenceTransformer model.

    Returns:
        A TokenRegistry instance.
    """
    return TokenRegistry(sentence_transformer)


def create_token_registry_from_embeddings(all_embeddings_df) -> TokenRegistry:
    """
    Create a TokenRegistry by loading precomputed embeddings
    from a DataFrame (e.g. all_embeddings.csv).

    Tokens are identified by rows whose index starts and ends with "< >".

    Args:
        all_embeddings_df: DataFrame containing embeddings for both
                           recipes and special tokens.

    Returns:
        A TokenRegistry initialized with preloaded token embeddings.
    """
    registry = TokenRegistry(sentence_transformer=None)

    token_embeddings = {}

    for idx in all_embeddings_df.index:
        if isinstance(idx, str) and idx.startswith("<") and idx.endswith(">"):
            embedding = all_embeddings_df.loc[idx].values.astype(np.float32)
            token_embeddings[idx] = embedding

    if token_embeddings:
        registry.set_embeddings_from_dict(token_embeddings)
        print(f"✅ TokenRegistry loaded with {len(token_embeddings)} tokens from embeddings")
    else:
        print("⚠️ No token found in all_embeddings.csv")

    return registry


def is_token(value) -> bool:
    """Return True if the value is a special token."""
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def is_meal_token(value) -> bool:
    """Return True if the token represents a meal."""
    return value in TOKEN_TO_MEAL


def is_course_token(value) -> bool:
    """Return True if the token represents a course type."""
    return value in TOKEN_TO_COURSE


def get_token_display(token: str) -> str:
    """Return a friendly emoji label for the given token."""
    return TOKEN_DISPLAY.get(token, token)
