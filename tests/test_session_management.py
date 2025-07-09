#!/usr/bin/env python3
"""
Test script to validate session management behavior from user perspective
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from modules.session import SessionManager


def test_user_chat_session_workflow():
    """Test realistic user chat workflow with model changes"""
    print("Testing realistic user chat workflow...")

    # Create session manager with realistic config
    config = {
        "chat": {
            "max_history_length": 20,
            "max_message_length": 10000,
            "session_timeout_hours": 24,
        }
    }

    sm = SessionManager(config)

    # Simulate user starting a chat with Model A
    print("\n1. User starts chat with qwen3_235b")
    session = sm.get_or_create_session(
        session_id="user_session_1", model_key="qwen3_235b", session_type="single"
    )

    # User has conversation with Model A
    print("2. User has conversation with qwen3_235b")
    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."},
        {"role": "user", "content": "Can you give me an example?"},
        {"role": "assistant", "content": "Sure! Here's a simple example..."},
    ]

    sm.set_conversation_history("user_session_1", messages)
    history = sm.get_conversation_history("user_session_1")

    print(f"   Conversation history length: {len(history)}")
    print(f"   Last message: {history[-1]['content'][:50]}...")

    # User switches to Model B - should they see previous conversation?
    print("\n3. User switches to llama_scout (different model)")
    session = sm.get_or_create_session(
        session_id="user_session_1", model_key="llama_scout", session_type="single"
    )

    # Check what conversation history is available
    history_after_switch = sm.get_conversation_history("user_session_1")
    print(
        f"   Conversation history after model switch: {len(history_after_switch)} messages"
    )

    # Expected behavior: History should be cleared for fresh start with new model
    if len(history_after_switch) == 0:
        print("   ✓ CORRECT: History cleared - user gets fresh start with new model")
    else:
        print(
            "   ✗ INCORRECT: History preserved - user sees old conversation with different model"
        )
        print(f"   Previous conversation: {history_after_switch}")
        return False

    # User continues with Model B
    print("\n4. User continues conversation with llama_scout")
    new_messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    ]

    sm.set_conversation_history("user_session_1", new_messages)
    history = sm.get_conversation_history("user_session_1")

    print(f"   New conversation history length: {len(history)}")
    print(f"   Current model: {session.model_key}")

    # User switches back to Model A - should get fresh start again
    print("\n5. User switches back to qwen3_235b")
    session = sm.get_or_create_session(
        session_id="user_session_1", model_key="qwen3_235b", session_type="single"
    )

    history_after_switch_back = sm.get_conversation_history("user_session_1")
    print(
        f"   Conversation history after switching back: {len(history_after_switch_back)} messages"
    )

    # Expected behavior: Fresh start again
    if len(history_after_switch_back) == 0:
        print("   ✓ CORRECT: Fresh start when switching back to original model")
    else:
        print("   ✗ INCORRECT: Should get fresh start when switching back")
        return False

    return True


def test_comparison_chat_workflow():
    """Test comparison chat workflow with model changes"""
    print("\n\nTesting comparison chat workflow...")

    config = {
        "chat": {
            "max_history_length": 20,
            "max_message_length": 10000,
            "session_timeout_hours": 24,
        }
    }

    sm = SessionManager(config)

    # User starts comparison between Model A and Model B
    print("\n1. User starts comparison: qwen3_235b vs llama_scout")
    session = sm.get_or_create_session(
        session_id="comparison_session_1",
        model_keys=["qwen3_235b", "llama_scout"],
        session_type="compare",
    )

    # User has conversation with both models
    print("2. User has conversation with both models")
    messages = [
        {"role": "user", "content": "Compare Python and JavaScript"},
        {"role": "assistant", "content": "Python is better for data science..."},
        {"role": "user", "content": "What about web development?"},
        {"role": "assistant", "content": "JavaScript is essential for web dev..."},
    ]

    sm.set_conversation_history("comparison_session_1", messages)
    history = sm.get_conversation_history("comparison_session_1")

    print(f"   Conversation history length: {len(history)}")
    print(f"   Comparing: {session.model_keys}")

    # User changes comparison to Model B vs Model C
    print("\n3. User changes comparison to llama_scout vs deepseek_r1")
    session = sm.get_or_create_session(
        session_id="comparison_session_1",
        model_keys=["llama_scout", "deepseek_r1"],
        session_type="compare",
    )

    history_after_change = sm.get_conversation_history("comparison_session_1")
    print(
        f"   Conversation history after model change: {len(history_after_change)} messages"
    )
    print(f"   Now comparing: {session.model_keys}")

    # Expected behavior: History should be cleared for fair comparison
    if len(history_after_change) == 0:
        print("   ✓ CORRECT: History cleared - fair comparison with new model pair")
    else:
        print("   ✗ INCORRECT: History preserved - unfair comparison (old context)")
        return False

    # User keeps same comparison (should preserve history)
    print("\n4. User continues with same comparison")
    new_messages = [
        {"role": "user", "content": "Which is better for AI development?"},
        {"role": "assistant", "content": "Both have their strengths..."},
    ]

    sm.set_conversation_history("comparison_session_1", new_messages)

    # Get session again with same models
    session = sm.get_or_create_session(
        session_id="comparison_session_1",
        model_keys=["llama_scout", "deepseek_r1"],
        session_type="compare",
    )

    history_same_models = sm.get_conversation_history("comparison_session_1")
    print(
        f"   Conversation history with same models: {len(history_same_models)} messages"
    )

    # Expected behavior: History should be preserved
    if len(history_same_models) == 2:
        print("   ✓ CORRECT: History preserved when models don't change")
    else:
        print("   ✗ INCORRECT: History should be preserved when models are same")
        return False

    return True


def test_session_persistence():
    """Test that sessions persist correctly across multiple requests"""
    print("\n\nTesting session persistence...")

    config = {
        "chat": {
            "max_history_length": 20,
            "max_message_length": 10000,
            "session_timeout_hours": 24,
        }
    }

    sm = SessionManager(config)

    # Create session
    session1 = sm.get_or_create_session(
        session_id="persistent_session", model_key="qwen3_235b", session_type="single"
    )

    # Add messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    sm.set_conversation_history("persistent_session", messages)

    # Get session again (should be same object)
    session2 = sm.get_or_create_session(
        session_id="persistent_session", model_key="qwen3_235b", session_type="single"
    )

    # Check persistence
    if session1 is session2:
        print("   ✓ CORRECT: Same session object returned")
    else:
        print("   ✗ INCORRECT: Different session objects")
        return False

    history = sm.get_conversation_history("persistent_session")
    if len(history) == 2:
        print("   ✓ CORRECT: Conversation history persisted")
    else:
        print("   ✗ INCORRECT: History not persisted")
        return False

    return True


if __name__ == "__main__":
    print("Testing Session Management from User Perspective")
    print("=" * 60)

    try:
        success = True

        success &= test_user_chat_session_workflow()
        success &= test_comparison_chat_workflow()
        success &= test_session_persistence()

        print("\n" + "=" * 60)
        if success:
            print(
                "✓ All tests passed! Session management behaves correctly from user perspective."
            )
        else:
            print("✗ Some tests failed. Session management needs fixes.")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
