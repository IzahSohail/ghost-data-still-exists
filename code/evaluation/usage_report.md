# Token usage and cost – final full-dataset run

Run: `python main.py` from `code/`, producing `output.csv` with 518 rows.

## Models configured

| Provider | Model | Used for |
| --- | --- | --- |
| Google | gemini-2.5-flash | Deciding whether a request text authorises deletion, only when the rule-based classifier returns no decision |

No other model is configured. Evidence parsing, identity resolution, state
reconstruction, policy application and step ordering are plain Python.

## Calls in the final run

| Metric | Value |
| --- | --- |
| Requests processed | 518 |
| Model calls | 0 |
| Input tokens | 0 |
| Output tokens | 0 |
| Total tokens | 0 |
| Average tokens per request | 0 |
| Estimated total cost | $0.00 |
| Estimated cost per request | $0.00 |

The rule-based classifier (`classify_request_by_rules` in `main.py`) decided
all 518 request texts (508 deletion, 10 not-deletion), so the model fallback was
never reached. The run prints these counters at the end
(`llm usage: {...}`); the numbers above are copied from that output.

## If the fallback is used

Each fallback call sends one request text with a fixed prompt (about 180
tokens of instructions plus the text) and asks for a JSON object of two
fields (about 30 output tokens). At the published gemini-2.5-flash list price
(USD 0.30 per million input tokens, USD 2.50 per million output tokens) one
call costs roughly USD 0.0002. Results are cached per normalised text, so
repeated wording is charged once.
