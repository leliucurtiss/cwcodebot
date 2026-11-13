# RAG Enhancement Plan: Hierarchical Contextual Retrieval (HCR)

## Goal
Transform the `ContextBuilder` from a simple sliding-window buffer into a retrieval-augmented system that balances global project state, local conversation state, and specific historical knowledge.

## Implementation Phases

### Phase 1: Keyword-Based History Retrieval
**Objective:** Stop relying solely on the most recent messages. Retrieve historical messages that are relevant to the current query.

- [ ] **Implement `_retrieve_relevant_history`**:
    - Extract keywords from the latest user message.
    - Scan `stored_messages` for these keywords.
    - Rank messages by a combination of relevance (keyword match count) and recency.
- [ ] **Integrate into `build()`**:
    - Create a `relevant_history_bucket`.
    - Fill this bucket before the `recent_history_bucket`.

### Phase 2: Enhanced Memory Integration
**Objective:** Ensure the Project and Conversation summaries are utilized effectively as the "Global" and "Local" state.

- [ ] **Refine `HierarchicalMemory.render()`**:
    - Improve the formatting to clearly distinguish between Project-level constraints and Conversation-level goals.
- [ ] **Update `ContextBuilder.build()`**:
    - Ensure the system prompt is constructed to treat these memories as "Ground Truth" for the session.

### Phase 3: Budget & Priority Optimization
**Objective:** Implement a strict priority system to ensure critical context is never discarded.

- [ ] **Define Priority Order**:
    1. System Prompt (Project/Conv Memory)
    2. Current Turn (Incoming Messages)
    3. Relevant History (RAG results)
    4. Recent History (Temporal window)
    5. Explicit Files
- [ ] **Implement "Soft" vs "Hard" Limits**:
    - Use `safe_input_budget` as a hard limit.
    - Implement a mechanism to truncate the lowest priority bucket first.

### Phase 4: Implicit File Discovery (Stretch Goal)
**Objective:** Automatically include files mentioned in the conversation.

- [ ] **Implement `_discover_implicit_files`**:
    - Use regex to find file paths or filenames in the current query.
    - Cross-reference with the workspace to find the actual relative path.
    - Add these to the `file_bucket`.

## Success Criteria
- The bot can answer questions about decisions made early in a long conversation.
- The bot maintains project-wide constraints (from Project Memory) across different conversations.
- Token budget is respected without cutting off the current user query.
