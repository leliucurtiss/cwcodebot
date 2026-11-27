from typing import Any, Tuple, List, Optional
from .models import ChatMessage
from .context import message_content_to_text, detect_repetition

async def prepare_chat_context(
    session_manager: Any,
    memory_manager: Any,
    storage: Any,
    context_builder: Any,
    logger: Any,
    messages: List[ChatMessage],
    project_id: Optional[str],
    conversation_id: Optional[str],
    context_files: Optional[List[str]] = None,
) -> Tuple[Any, Any, List[dict[str, str]]]:
    """
    Common logic to resolve session, load memory, build context, and save messages.
    Returns (session, payload_messages, original_messages).
    """
    session = session_manager.resolve(project_id, conversation_id)
    memory = memory_manager.load(session.project, session.conversation)
    stored = storage.list_messages(session.conversation.id)
    
    try:
        built = context_builder.build(
            session.project, 
            memory, 
            stored, 
            messages, 
            context_files or []
        )
    except ValueError as e:
        logger.error(f"Context building error: {e}")
        raise

    # Circuit Breaker: Detect repetition in the built context
    is_repeated, reason = detect_repetition(built.messages)
    if is_repeated:
        logger.debug(f"[Circuit Breaker] Repetition detected! Reason: {reason}. Injecting warning.")
        built.messages.append({
            "role": "system", 
            "content": "Warning: You are repeating yourself. Stop performing the same action and try a different approach or summarize your findings."
        })

    for msg in messages:
        if msg.role in {"user", "assistant"}:
            content = message_content_to_text(msg.content)
            # Avoid duplicating messages already in storage
            if not any(m.role == msg.role and m.content == content for m in stored):
                storage.add_message(session.conversation.id, msg.role, content)
    
    return session, built.messages, messages

def print_messages_length(messages, label):
    total_char_size = 0
    for i, message in enumerate(messages):
        # Handle the case where content is a list or string
        if isinstance(message.content, str):
            message_size = len(message.content)
        elif isinstance(message.content, list):
            message_size = sum(len(str(item)) for item in message.content)
        else:
            message_size = 0

        total_char_size += message_size

    # Print Total requests total character size
    # print(f"INFO: Total {label}s: {len(messages)} with {total_char_size} chars")
