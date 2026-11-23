import pytest
from local_ai_harness.context import detect_repetition

def test_detect_repetition_true():
    # Two very similar messages
    messages = [
        {"role": "assistant", "content": "I will read the rest of the file app.py to understand the logic."},
        {"role": "assistant", "content": "I will read the rest of the file app.py to understand the logic."}
    ]
    assert detect_repetition(messages) is True

def test_detect_repetition_false():
    # Two different messages
    messages = [
        {"role": "assistant", "content": "I have read the file app.py."},
        {"role": "assistant", "content": "Now I will check the tests in test_app.py."}
    ]
    assert detect_repetition(messages) is False

def test_detect_repetition_insufficient_messages():
    # Only one assistant message
    messages = [
        {"role": "assistant", "content": "Hello!"}
    ]
    assert detect_repetition(messages) is False
