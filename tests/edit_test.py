from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
from collections import OrderedDict

# --- MOCK CLASSES TO SIMULATE THE HARNESS ---

@dataclass
class ContextConfig:
    safe_input_budget: int = 1000
    token_char_ratio: float = 4.0
    token_safety_multiplier: float = 1.1
    max_compressed_length: int = 500

@dataclass
class Project:
    id: str
    memory_summary: str

@dataclass
class StoredMessage:
    role: str
    content: str

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class HierarchicalMemory:
    project_summary: str
    conversation_summary: str

    def render(self) -> str:
        return (
            f"PROJECT MEMORY:\n{self.project_summary or '(empty)'}\n\n"
            f"CONVERSATION MEMORY:\n{self.conversation_summary or '(empty)'}"
        )

@dataclass
class ContextBuildResult:
    messages: List[Dict[str, str]]
    estimated_tokens: int
    discarded: List[str] = field(default_factory=list)

# --- THE HCR IMPLEMENTATION ---

class HCRContextBuilder:
    def __init__(self, config: ContextConfig):
        self.config = config

    def estimate_tokens(self, text: str) -> int:
        if not text: return 0
        return int((len(text) / self.config.token_char_ratio) * self.config.token_safety_multiplier) + 1

    def _message_tokens(self, msg: Dict[str, str]) -> int:
        return self.estimate_tokens(msg.get("role", "")) + self.estimate_tokens(msg.get("content", ""))

    def _extract_keywords(self, text: str) -> set[str]:
        # Simple keyword extraction: words longer than 3 chars
        return {word.lower() for word in text.split() if len(word) > 3}

    def build(
        self,
        project: Project,
        conv_summary: str,
        stored_messages: List[StoredMessage],
        incoming_messages: List[ChatMessage],
    ) -> ContextBuildResult:
        budget = self.config.safe_input_budget
        used = 0
        discarded = []
        
        # Buckets for priority
        system_bucket = []
        incoming_bucket = []
        relevant_bucket = []
        recent_bucket = []

        def add_to_bucket(bucket, label, role, content, required=False):
            nonlocal used
            msg = {"role": role, "content": content}
            cost = self._message_tokens(msg)
            
            if not required and (used + cost > budget):
                discarded.append(label)
                return False
            
            bucket.append(msg)
            used += cost
            return True

        # 1. Hierarchical Memory (Highest Priority)
        mem = HierarchicalMemory(project.memory_summary, conv_summary)
        add_to_bucket(system_bucket, "system_mem", "system", mem.render(), required=True)

        # 2. Incoming Messages (High Priority)
        for msg in incoming_messages:
            add_to_bucket(incoming_bucket, "incoming", msg.role, msg.content, required=True)

        # 3. Relevance Retrieval (RAG Layer)
        # Use the last user message to find relevant history
        last_user_msg = next((m.content for m in reversed(incoming_messages) if m.role == "user"), "")
        keywords = self._extract_keywords(last_user_msg)
        
        if keywords:
            for sm in stored_messages:
                if any(kw in sm.content.lower() for kw in keywords):
                    # Avoid duplicating messages that are already in the 'recent' window (last 5)
                    if sm not in stored_messages[-5:]:
                        if not add_to_bucket(relevant_bucket, "relevant_hist", sm.role, sm.content):
                            break

        # 4. Recent History (Fill remaining budget)
        for sm in reversed(stored_messages):
            # Avoid duplicating what we already put in relevant_bucket
            if any(sm.content == r["content"] for r in relevant_bucket):
                continue
            if not add_to_bucket(recent_bucket, "recent_hist", sm.role, sm.content):
                break

        # Assemble: System -> Relevant -> Recent (reversed back to chrono) -> Incoming
        final_messages = system_bucket + list(reversed(recent_bucket)) + relevant_bucket + incoming_bucket
        
        return ContextBuildResult(
            messages=final_messages,
            estimated_tokens=sum(self._message_tokens(m) for m in final_messages),
            discarded=discarded
        )

class SandwichContextBuilder:
    """
    Baseline implementation to fight drift.
    Protects the 'Top Anchor' (Goal) and 'Bottom Anchor' (State),
    discarding the middle to break repetition loops.
    """
    def __init__(self, config: ContextConfig):
        self.config = config

    def estimate_tokens(self, text: str) -> int:
        if not text: return 0
        return int((len(text) / self.config.token_char_ratio) * self.config.token_safety_multiplier) + 1

    def _message_tokens(self, msg: Dict[str, str]) -> int:
        return self.estimate_tokens(msg.get("role", "")) + self.estimate_tokens(msg.get("content", ""))

    def build(
        self,
        project: Project,
        conv_summary: str,
        stored_messages: List[StoredMessage],
        incoming_messages: List[ChatMessage],
    ) -> ContextBuildResult:
        budget = self.config.safe_input_budget
        used = 0
        discarded = []
        final_messages = []

        # 1. Top Anchor: System Memory (Required)
        mem = HierarchicalMemory(project.memory_summary, conv_summary)
        sys_msg = {"role": "system", "content": mem.render()}
        used += self._message_tokens(sys_msg)
        final_messages.append(sys_msg)

        # 2. Top Anchor: The first few messages (The Goal)
        # We keep the first 2 messages to ensure the model remembers the original intent
        goal_anchor_size = 2
        for sm in stored_messages[:goal_anchor_size]:
            msg = {"role": sm.role, "content": sm.content}
            cost = self._message_tokens(msg)
            if used + cost <= budget:
                final_messages.append(msg)
                used += cost
            else:
                discarded.append(f"goal_{sm.role}")

        # 3. Bottom Anchor: The most recent messages (The State)
        # Fill remaining budget from the end backwards
        recent_messages = []
        for sm in reversed(stored_messages[goal_anchor_size:]):
            msg = {"role": sm.role, "content": sm.content}
            cost = self._message_tokens(msg)
            if used + cost <= budget:
                recent_messages.append(msg)
                used += cost
            else:
                discarded.append(f"recent_{sm.role}")
                break
        
        # Add recent messages in chronological order
        final_messages.extend(reversed(recent_messages))

        # 4. Incoming Messages (Required)
        for msg in incoming_messages:
            m = {"role": msg.role, "content": msg.content}
            cost = self._message_tokens(m)
            # If we are over budget, prune the bottom anchor to make room
            while used + cost > budget and recent_messages:
                removed = recent_messages.pop(0) # Remove oldest of the recent
                used -= self._message_tokens(removed)
                discarded.append("pruned_recent")
            
            final_messages.append(m)
            used += cost

        return ContextBuildResult(
            messages=final_messages,
            estimated_tokens=used,
            discarded=discarded
        )

def detect_repetition(messages: List[Dict[str, str]], threshold=0.7) -> bool:
    """
    Circuit Breaker: Detects if the assistant is repeating itself.
    Compares the last two assistant messages using Jaccard similarity.
    """
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    if len(assistant_msgs) < 2:
        return False
    
    m1 = assistant_msgs[-1].lower()
    m2 = assistant_msgs[-2].lower()
    
    set1 = set(m1.split())
    set2 = set(m2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    similarity = intersection / union if union > 0 else 0
    return similarity > threshold


# --- TEST SUITE ---

def test_hcr():
    config = ContextConfig(safe_input_budget=1500) # Moderate budget
    builder = HCRContextBuilder(config)
    
    project = Project("proj-1", "A high-performance trading engine written in Rust.")
    conv_summary = "Currently debugging the order matching logic."
    
    # History: Some old relevant stuff, some noise, some recent stuff
    history = [
        StoredMessage("user", "How does the matching engine handle limit orders?"),
        StoredMessage("assistant", "Limit orders are stored in a priority queue based on price."),
        StoredMessage("user", "What is the weather in London?"), # Noise
        StoredMessage("assistant", "I don't know the weather."), # Noise
        StoredMessage("user", "Let's talk about the database schema."),
        StoredMessage("assistant", "The schema uses PostgreSQL with a JSONB column for metadata."),
        StoredMessage("user", "I think the limit order logic is slow."), # Recent
        StoredMessage("assistant", "We should check the priority queue implementation."), # Recent
    ]
    
    # Current turn: User asks about limit orders again
    incoming = [
        ChatMessage("user", "Wait, you mentioned limit orders earlier, how exactly is the priority queue sorted?")
    ]
    
    print("--- Running HCR Build ---")
    result = builder.build(project, conv_summary, history, incoming)
    print(f"Estimated Tokens: {result.estimated_tokens}")
    print(f"Discarded: {result.discarded}")
    print("\nFinal Context Sequence:")
    for i, msg in enumerate(result.messages):
        print(f"{i}. [{msg['role']}] {msg['content'][:60]}...")

def test_sandwich():
    config = ContextConfig(safe_input_budget=800) # Tight budget to force pruning
    builder = SandwichContextBuilder(config)
    
    project = Project("proj-1", "A high-performance trading engine.")
    conv_summary = "Debugging order matching."
    
    # History: Goal -> Noise/Drift -> Recent State
    history = [
        StoredMessage("user", "Goal: Implement a limit order book."), # Goal Anchor
        StoredMessage("assistant", "I will help you implement the order book."), # Goal Anchor
        StoredMessage("user", "Wait, what is 2+2?"), # Noise
        StoredMessage("assistant", "2+2 is 4."), # Noise
        StoredMessage("user", "Tell me a joke."), # Noise
        StoredMessage("assistant", "Why did the chicken cross the road?"), # Noise
        StoredMessage("user", "Back to the order book, how do we sort?"), # Recent State
        StoredMessage("assistant", "We use a B-Tree for sorting."), # Recent State
    ]
    
    incoming = [ChatMessage("user", "Can you explain the B-Tree part again?")]
    
    print("\n--- Running Sandwich Build ---")
    result = builder.build(project, conv_summary, history, incoming)
    print(f"Estimated Tokens: {result.estimated_tokens}")
    print(f"Discarded: {result.discarded}")
    print("\nFinal Context Sequence (Notice the middle is gone):")
    for i, msg in enumerate(result.messages):
        print(f"{i}. [{msg['role']}] {msg['content'][:60]}...")

    # Test Repetition Detection
    drift_history = [
        {"role": "assistant", "content": "The B-Tree is a self-balancing tree data structure."},
        {"role": "assistant", "content": "The B-Tree is a self-balancing tree data structure."}
    ]
    print(f"\nRepetition Detected: {detect_repetition(drift_history)}")

if __name__ == "__main__":
    test_hcr()
    test_sandwich()