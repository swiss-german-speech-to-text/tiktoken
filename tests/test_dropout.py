"""
Tests for the dropout_prob parameter in BPE encoding (tiktoken.encode and _encode_bytes).

These tests verify that for a given encoding (gpt-4), a dropout_prob of 0.0
matches the default behavior, and a dropout_prob of 1.0 produces a different,
repeatable tokenization compared to the default, for both text and byte inputs.
"""
import pytest
import tiktoken

# Skip all tests in this module if the gpt-4 encoding (cl100k_base) cannot be loaded,
# e.g., due to missing network access to download encoding files.
try:
    _ = tiktoken.encoding_for_model("gpt-4")
except Exception:
    pytest.skip(
        "Unable to load gpt-4 encoding (network required); skipping dropout tests",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "text",
    [
        "SwissAir",
        "Vincenzo Timmel",
    ],
)
def test_dropout_probability_zero_equivalent_to_default(text: str):
    """
    A dropout_prob of 0.0 should match the default encoding behavior.
    """
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens_default = enc.encode(text)
    tokens_no_dropout = enc.encode(text, dropout_prob=0.0)
    assert tokens_no_dropout == tokens_default


@pytest.mark.parametrize(
    "text",
    [
        "SwissAir",
        "Vincenzo Timmel",
    ],
)
def test_dropout_probability_one_is_repeatable_and_differs_from_default(text: str):
    """
    A dropout_prob of 1.0 should break BPE merges on the first step (no merges),
    yielding a deterministic but different tokenization from the default.
    """
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens_default = enc.encode(text)
    tokens_dropout1 = enc.encode(text, dropout_prob=1.0)
    tokens_dropout2 = enc.encode(text, dropout_prob=1.0)
    # Dropout at 100% should be deterministic (no randomness) but differ from default
    assert tokens_dropout1 == tokens_dropout2
    assert tokens_dropout1 != tokens_default


@pytest.mark.parametrize(
    "text",
    [
        "SwissAir",
        "Vincenzo Timmel",
    ],
)
def test_dropout_probability_zero_equivalent_to_default_bytes(text: str):
    """
    A dropout_prob of 0.0 for bytes input should match the default byte-level encoding behavior.
    """
    enc = tiktoken.encoding_for_model("gpt-4")
    data = text.encode("utf-8")
    tokens_default = enc._encode_bytes(data)
    tokens_no_dropout = enc._encode_bytes(data, dropout_prob=0.0)
    assert tokens_no_dropout == tokens_default


@pytest.mark.parametrize(
    "text",
    [
        "SwissAir",
        "Vincenzo Timmel",
    ],
)
def test_dropout_probability_one_is_repeatable_and_differs_from_default_bytes(text: str):
    """
    A dropout_prob of 1.0 for bytes input should break BPE merges on the first step (no merges),
    yielding a deterministic but different tokenization from the default.
    """
    enc = tiktoken.encoding_for_model("gpt-4")
    data = text.encode("utf-8")
    tokens_default = enc._encode_bytes(data)
    tokens_dropout1 = enc._encode_bytes(data, dropout_prob=1.0)
    tokens_dropout2 = enc._encode_bytes(data, dropout_prob=1.0)
    assert tokens_dropout1 == tokens_dropout2
    assert tokens_dropout1 != tokens_default