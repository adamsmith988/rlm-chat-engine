---
name: remember
description: Store and recall durable facts the user asked the agent to keep.
when_to_use: When the user states a preference, name, constraint, or fact that must survive later turns and compaction.
---

# Remember

Use this skill whenever the user asks you to remember something, or states a durable fact.

## Steps

1. Put the fact in a named Python variable.
2. Persist it in the continual harness:

```python
await rlm.harness.upsert_memory(
    content="<the fact in the user's own words>",
    title="<short title>",
)
```

3. Confirm back to the user, in their language, what you stored.
4. On later turns, read `await rlm.harness.list(kind="memory")` before answering questions about stored facts. Do not invent facts that are not in harness or recalled memory.
