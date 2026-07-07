# Dispatch prompt template

Fill the `{SLOTS}` and send as the content-reviewer dispatch prompt, verbatim. Add nothing — this template is the caller-side enforcement of the blinding: there is no slot for background, intent, or steering, and none may be added.

```
Blind content review.

Artifact type: {ARTIFACT_TYPE}

Artifact set (review exactly these paths, nothing else):
{PATHS_ONE_PER_LINE}

Write your report to: {REPORT_PATH}

No further context is provided. That is deliberate — you are reading cold.
```
