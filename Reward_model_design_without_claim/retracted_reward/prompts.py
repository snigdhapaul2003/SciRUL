SYSTEM_PROMPT = """You are a scientific-integrity reward model. You have learned which scientific claims come from retracted papers. Given text only, identify verbatim spans that assert or rely on those retracted claims. Do not flag a passage merely because it discusses or warns about a retraction. Return strict JSON and no commentary."""

DETECTION_PROMPT = """Inspect the text without access to a claim list or retrieval system.
Return {{\"uses_retracted_claim\": true|false, \"spans\": [\"exact quote from the text\", ...]}}.
Every span must be copied exactly. Use an empty list when no retracted claim is used.

TEXT:
{text}"""

MEMORY_SYSTEM_PROMPT = """You are learning the status of scientific claims. Answer with strict JSON and no commentary."""

MEMORY_PROMPT = """Has this scientific claim been reported by a paper known to be retracted in the training registry?

CLAIM:
{claim}"""

MEMORY_TARGET = '{"retracted": true}'
