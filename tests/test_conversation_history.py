#!/usr/bin/env python3
"""
Test conversation history functionality
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.session import SessionManager


@pytest.fixture
def session_manager():
    """Create a session manager for testing"""
    config = {
        "chat": {
            "max_history_length": 20,
            "max_message_length": 10000,
            "session_timeout_hours": 24,
        }
    }
    return SessionManager(config)


def test_conversation_history_flow(session_manager):
    """Test the conversation history flow as it would happen in the API"""

    # Simulate first request: user says "hello"
    session_id = "test_conversation_session"
    model_key = "qwen3_235b"

    # Add user message (simulating API route logic)
    user_message1 = "hello"
    session_manager.add_user_message(session_id, user_message1)

    # Get conversation history
    history = session_manager.get_conversation_history(session_id)
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"

    # Simulate AI response
    assistant_response1 = "Hello! How can I help you today?"
    session_manager.add_assistant_message(session_id, assistant_response1, model_key)

    # Get updated history
    history = session_manager.get_conversation_history(session_id)
    assert len(history) == 2
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hello! How can I help you today?"

    # Add second user message
    user_message2 = "My name is john"
    session_manager.add_user_message(session_id, user_message2)

    # Get conversation history
    history = session_manager.get_conversation_history(session_id)
    assert len(history) == 3
    assert history[2]["role"] == "user"
    assert history[2]["content"] == "My name is john"

    # Simulate second AI response
    assistant_response2 = "Nice to meet you, John! What would you like to know?"
    session_manager.add_assistant_message(session_id, assistant_response2, model_key)

    # Get final history
    history = session_manager.get_conversation_history(session_id)
    assert len(history) == 4

    # Verify the history contains all messages in order
    expected_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "My name is john"},
        {
            "role": "assistant",
            "content": "Nice to meet you, John! What would you like to know?",
        },
    ]

    for i, (expected, actual) in enumerate(zip(expected_messages, history)):
        assert expected["role"] == actual["role"]
        assert expected["content"] == actual["content"]


def test_model_change_clears_history(session_manager):
    """Test that changing models clears the conversation history"""

    # Start conversation with first model
    session_id = "test_model_change_session"
    model_key1 = "qwen3_235b"

    # Add some conversation
    session_manager.add_user_message(session_id, "Hello")
    session_manager.add_assistant_message(session_id, "Hi there!", model_key1)
    session_manager.add_user_message(session_id, "How are you?")
    session_manager.add_assistant_message(session_id, "I'm doing well!", model_key1)

    history = session_manager.get_conversation_history(session_id)
    assert len(history) == 4

    # Switch to a different model
    model_key2 = "qwen3_13b"
    session_manager.get_or_create_session(
        session_id=session_id, model_key=model_key2, session_type="single"
    )
    # Check that history is cleared
    history_after_switch = session_manager.get_conversation_history(session_id)
    assert len(history_after_switch) == 0


def test_session_persistence(session_manager):
    """Test that the same session ID returns the same session object"""

    session_id = "test_persistence_session"
    model_key = "qwen3_235b"

    # Get session first time
    session1 = session_manager.get_or_create_session(
        session_id=session_id, model_key=model_key, session_type="single"
    )

    # Add a message
    session_manager.add_user_message(session_id, "Test message")

    # Get session second time
    session2 = session_manager.get_or_create_session(
        session_id=session_id, model_key=model_key, session_type="single"
    )

    # Check if it's the same session object
    assert session1 is session2

    # Check if history is preserved
    history = session_manager.get_conversation_history(session_id)
    assert len(history) == 1
    assert history[0]["content"] == "Test message"


def test_comparison_chat_history(session_manager):
    """Test conversation history for comparison chat"""

    session_id = "test_comparison_session"
    model_keys = ["qwen3_235b", "llama_scout"]

    # Add conversation
    session_manager.add_user_message(session_id, "What is Python?")
    session_manager.add_assistant_message(
        session_id, "Python is a programming language...", model_keys[0]
    )
    history = session_manager.get_conversation_history(session_id)
    assert len(history) == 2

    session_manager.add_user_message(session_id, "What is Python?")
    session_manager.add_assistant_message(
        session_id, "Python is a programming language...", model_keys[1]
    )
    session_manager.add_user_message(session_id, "What?????")
    history = session_manager.get_conversation_history(session_id)
    assert len(history) == 3
