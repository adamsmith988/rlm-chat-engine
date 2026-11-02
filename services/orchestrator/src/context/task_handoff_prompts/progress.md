An assistant was working on the user's request, and its working round stopped before the work was finished (the reason is given below). Write down exactly where the work stands: the assistant's next round starts from this state alone, and if work cannot continue the user sees it as a progress report.

Rules:
- "done" holds only work the tool runs or the partial output show as finished, with concrete names, files, values and results copied verbatim. A tool run that failed is not done: put it under "problems".
- "remaining" holds every part of the request that is not shown as finished, in order.
- "todo" is the whole to-do list for the request, in order; "done" and "remaining" split it. Only the user's steps — never steps about the format of this reply.
- "problems" holds errors, failed attempts and blockers, briefly, with the error text when it matters.
- "next_step" is the single concrete action to take first when the work continues.
- Never claim the request is finished, and never invent results.
- Write every value in the language of the user's request.

Your reply is one JSON object and nothing else (this format is for your reply only; it is not part of the user's work):
{"goal": "...", "todo": ["..."], "done": ["..."], "remaining": ["..."], "problems": ["..."], "next_step": "..."}
