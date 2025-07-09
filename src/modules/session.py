import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from src.logger import logger


@dataclass
class ConversationSession:
    """Represents a conversation session with message history and metadata."""

    session_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    model_key: Optional[str] = None
    model_keys: Optional[List[str]] = None
    session_type: str = "single"  # "single" or "compare"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(message)
        self.last_activity = time.time()

    def get_conversation_history(
        self, max_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history with optional length limit."""
        if max_length is None:
            return self.conversation_history.copy()

        if max_length <= 0:
            return []

        # Keep system messages and most recent messages
        system_messages = [
            msg for msg in self.conversation_history if msg.get("role") == "system"
        ]

        if len(system_messages) >= max_length:
            return system_messages[:max_length]

        # Get recent messages while preserving system messages
        recent_messages = self.conversation_history[
            -(max_length - len(system_messages)) :
        ]

        # Combine system messages with recent messages, avoiding duplicates
        result = system_messages.copy()
        for msg in recent_messages:
            if msg not in result:
                result.append(msg)

        return result

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.conversation_history.clear()
        self.last_activity = time.time()

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = time.time()

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if the session has expired."""
        return (time.time() - self.last_activity) > (max_age_hours * 3600)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "model_key": self.model_key,
            "model_keys": self.model_keys,
            "session_type": self.session_type,
            "metadata": self.metadata,
        }


class SessionManager:
    """Manages conversation sessions with history tracking and cleanup."""

    def __init__(self, config: Dict[str, Any]):
        self.sessions: Dict[str, ConversationSession] = {}
        self.config = config
        self.max_history_length = config.get("chat", {}).get("max_history_length", 20)
        self.max_message_length = config.get("chat", {}).get(
            "max_message_length", 10000
        )
        self.session_timeout_hours = config.get("chat", {}).get(
            "session_timeout_hours", 24
        )

        logger.info(
            f"SessionManager initialized with max_history_length={self.max_history_length}"
        )

    def create_session(
        self,
        session_id: Optional[str] = None,
        model_key: Optional[str] = None,
        model_keys: Optional[List[str]] = None,
        session_type: str = "single",
    ) -> ConversationSession:
        """Create a new conversation session."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        session = ConversationSession(
            session_id=session_id,
            model_key=model_key,
            model_keys=model_keys,
            session_type=session_type,
        )

        self.sessions[session_id] = session
        logger.debug(f"Created new session: {session_id}, type: {session_type}")
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing session by ID."""
        return self.sessions.get(session_id)

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        model_key: Optional[str] = None,
        model_keys: Optional[List[str]] = None,
        session_type: str = "single",
    ) -> ConversationSession:
        """Get an existing session or create a new one. Clears history if model changes."""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]

            # Check if model has changed and clear history if needed
            model_changed = False
            if session_type == "single" and model_key is not None:
                if session.model_key != model_key:
                    model_changed = True
                    old_model = session.model_key
                    session.model_key = model_key
                    logger.info(
                        f"Model changed in session {session_id} from {old_model} to {model_key}"
                    )
            elif session_type == "compare" and model_keys is not None:
                if session.model_keys != model_keys:
                    model_changed = True
                    old_models = session.model_keys
                    session.model_keys = model_keys
                    logger.info(
                        f"Models changed in session {session_id} from {old_models} to {model_keys}"
                    )

            if model_changed:
                session.clear_history()
                logger.info(
                    f"Cleared conversation history for session {session_id} due to model change"
                )

            session.update_activity()
            return session

        return self.create_session(session_id, model_key, model_keys, session_type)

    def add_user_message(self, session_id: str, content: str) -> List[Dict[str, Any]]:
        """Add a user message to session and return conversation history."""
        session = self.get_or_create_session(session_id)

        # Validate message length
        if len(content) > self.max_message_length:
            logger.warning(
                f"Message truncated from {len(content)} to {self.max_message_length} characters"
            )
            content = content[: self.max_message_length]

        user_message = {"role": "user", "content": content}
        session.add_message(user_message)

        return session.get_conversation_history(self.max_history_length)

    def add_assistant_message(
        self, session_id: str, content: str, model_key: Optional[str] = None
    ) -> None:
        """Add an assistant message to session."""
        session = self.get_session(session_id)
        if not session:
            logger.warning(
                f"Attempted to add assistant message to non-existent session: {session_id}"
            )
            return

        # Validate message length
        if len(content) > self.max_message_length:
            logger.warning(
                f"Assistant message truncated from {len(content)} to {self.max_message_length} characters"
            )
            content = content[: self.max_message_length]

        assistant_message = {"role": "assistant", "content": content}

        session.add_message(assistant_message)

    def set_conversation_history(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> None:
        """Set the complete conversation history for a session."""
        session = self.get_or_create_session(session_id)

        # Validate and clean messages
        cleaned_messages = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                # Validate message length
                content = msg["content"]
                if len(content) > self.max_message_length:
                    logger.warning(
                        f"Message content truncated from {len(content)} to {self.max_message_length} characters"
                    )
                    content = content[: self.max_message_length]

                cleaned_message = {"role": msg["role"], "content": content}
                cleaned_messages.append(cleaned_message)

        # Apply history length limit
        if len(cleaned_messages) > self.max_history_length:
            logger.info(
                f"Conversation history truncated from {len(cleaned_messages)} to {self.max_history_length} messages"
            )
            cleaned_messages = cleaned_messages[-self.max_history_length :]

        session.conversation_history = cleaned_messages
        session.update_activity()

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        session = self.get_session(session_id)
        if not session:
            return []

        return session.get_conversation_history(self.max_history_length)

    def update_session_activity(self, session_id: str) -> None:
        """Update session activity timestamp."""
        session = self.get_session(session_id)
        if session:
            session.update_activity()

    def reset_session(self, session_id: str) -> None:
        """Reset session conversation history."""
        session = self.get_session(session_id)
        if session:
            session.clear_history()
            logger.debug(f"Reset session history: {session_id}")

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.debug(f"Deleted session: {session_id}")
            return True
        return False

    def cleanup_expired_sessions(self, max_age_hours: Optional[int] = None) -> int:
        """Clean up expired sessions and return count of removed sessions."""
        if max_age_hours is None:
            max_age_hours = self.session_timeout_hours

        expired_session_ids = []
        for session_id, session in self.sessions.items():
            if session.is_expired(max_age_hours):
                expired_session_ids.append(session_id)

        for session_id in expired_session_ids:
            del self.sessions[session_id]

        if expired_session_ids:
            logger.info(f"Cleaned up {len(expired_session_ids)} expired sessions")

        return len(expired_session_ids)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about current sessions."""
        active_sessions = len(self.sessions)
        single_sessions = sum(
            1 for s in self.sessions.values() if s.session_type == "single"
        )
        compare_sessions = sum(
            1 for s in self.sessions.values() if s.session_type == "compare"
        )

        total_messages = sum(
            len(s.conversation_history) for s in self.sessions.values()
        )

        return {
            "active_sessions": active_sessions,
            "single_sessions": single_sessions,
            "compare_sessions": compare_sessions,
            "total_messages": total_messages,
            "max_history_length": self.max_history_length,
            "max_message_length": self.max_message_length,
            "session_timeout_hours": self.session_timeout_hours,
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with basic information."""
        return [
            {
                "session_id": session.session_id,
                "session_type": session.session_type,
                "model_key": session.model_key,
                "model_keys": session.model_keys,
                "message_count": len(session.conversation_history),
                "created_at": session.created_at,
                "last_activity": session.last_activity,
            }
            for session in self.sessions.values()
        ]
