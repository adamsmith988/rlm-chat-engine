---
name: repl-inspect
description: Inspect the live IPython kernel instead of guessing at hidden state.
when_to_use: When you need to know which variables, files, or skill bodies exist after compaction or a long conversation.
---

# REPL inspect

Run this pattern in `ipython` before claiming the kernel is empty:

```python
st = skills.file_stats(".")
names = [n for n in dir() if not n.startswith("_")]
catalog = skills.list_markdown_skills()
catalog, names[:50], st
```

If a markdown skill is needed, load the full body with `skills.use("<name>")` and follow it completely before improvising.
