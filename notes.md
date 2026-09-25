# LangChain Master Mind Map — Prompting & Models

> The mental model for LangChain course Module 1 (foundational models + prompting).
> One goal: **control the model's answer**.

## Table of Contents (table des matières)

1. **Chatbot vs RAG vs Agentic vs Corrective RAG** — the 4 knowledge concepts (below)
2. The Master Map — 5 layers of control
3. The 4 moves (Worker / Post-it / Box / Read) — the execution skeleton
4. The 5 cages of prompting
5. The 2 doors of the system prompt
6. Structured output
7. Tools & Invocation (Module 2 preview)
8. How tools reach the model (the catalog)
9. The 7-step agent recipe
10. Mental model: `invoke` is a waiter (the tape & hidden loop)
11. LangChain × providers: string guessing vs. explicit pinning (AI Studio fix)
12. Tavily: ready-made tool vs. build your own
13. Multimodal: how images & audio become tokens
14. MCP: the USB-C of AI integrations
15. Managing long conversations (Middleware · Agent · Checkpointer · Store)
16. The wedding agent walkthrough + the SQL path fix
17. Python return types for chatbots (dict vs list)
18. Python's useful built-in errors (the common families)
19. The full request flow (client → API → logic → DB/LLM/API) + where errors appear
20. Agent memory: checkpointer + thread_id (remembering conversations)
21. The 4 token-saving families (+ Trim explained)
22. Working with JSON in Python, requests & Flask (the full reference)

## Chatbot vs RAG vs Agentic vs Corrective RAG

> *Chatbot **knows** (memory) · RAG **fetches** (your files) · Agent **decides** (tools) · Corrective RAG **fetches, then if unsure — searches**.*

| Concept | Where the answer's knowledge comes from | Who controls "what happens next" |
|---|---|---|
| **Chatbot (plain LLM)** | the model's **training weights** — frozen at cutoff | nobody — one answer |
| **RAG** | **your external corpus** (retrieved chunks stuffed into the prompt) | a **fixed pipeline** (retrieve → stuff → answer), not the model |
| **Agentic** | wherever tools fetch from (any tool's output) | the **model** — it decides which tool to call, in a loop |
| **Corrective RAG** (= agentic RAG) | **your index first; web fallback if the model thinks retrieval failed** | the **model** — it self-repairs a weak retrieval by calling `web_search` |

- **Chatbot**: answers from the data the LLM was trained on.
- **RAG**: retrieving from external data (your vector DB) and grounding the answer in it.
- **Agentic**: using tools — the model decides when/what to call (the `tool_calls` loop).
- **Corrective RAG**: retrieve first; if the index can't answer, go fetch from the web.

## The Master Map

```
                    ┌──────────────────────────────┐
                    │       THE ONE GOAL           │
                    │   "Control the model's answer" │
                    └──────────────────────────────┘

        ┌──────────────────┬──────────────┬─────────────────┐
        ▼                  ▼              ▼                 ▼
   LAYER 1           LAYER 2         LAYER 3          LAYER 4
 The Conversation    The 4 Moves    The 5 Cages      Where the prompt lives
 (the pipe)          (code shape)   (how to control) (the 2 doors)
```

## LAYER 1 — The Conversation (the only thing that exists)

```
messages = [ System?, Human, AIMessage, Human, AIMessage, ... ]

roles:   System = instructions   ·   Human = you   ·   AIMessage = the model
ORDER IS EVERYTHING → that's why it's a LIST (positions matter)
```

**How to read the answer (always the same):**

```python
response["messages"]   # the list of every turn
              [-1]     # last element = the model's final answer
              .text    # clean string (what you read)
              .content # same text but Gemini BLOCKS: [{'type':'text','text':'...'}]
```

## LAYER 2 — The 4 Moves (every single code block ever)

```python
# ① build the WORKER
agent = create_agent(model=...)

# ② write the POST-IT
question = HumanMessage("...")

# ③ exchange in the BOX
response = agent.invoke({"messages": [question]})

# ④ read the LAST LETTER
print(response["messages"][-1].text)
```

> Mnemonic: **Worker · Post-it · Box · Read** (French: *Ouvrier · Mot · Boîte · Réponse*)

## LAYER 3 — The 5 Cages (prompting = tightening the cage around freedom)

| # | Technique | The one line | Output type | Freedom |
|---|-----------|--------------|-------------|---------|
| ① | **Basic** | `create_agent(model)` | free text | ████ |
| ② | **Persona** | `system_prompt="you are a sci-fi writer"` | shaped text | ███ |
| ③ | **Few-shot** | examples "Marsialis" in the prompt | mimicked style | ██ |
| ④ | **Structured text** | `Name: Location: Vibe: Economy:` | labeled text | █ |
| ⑤ | **Structured output** | `class X(BaseModel)` + `response_format=X` | GUARANTEED data object (json) | ░ |

**The logic of the order (the "why"):**

```
① You say nothing  → ② You say who  → ③ You show examples
  → ④ You list the format  → ⑤ You FORCE the format

Control goes UP  ·  Model freedom goes DOWN  ·  Predictability goes UP
```

## LAYER 4 — The 2 Doors (where a system prompt lives)

**Door A: inside `create_agent` — fixed, every call**

```python
agent = create_agent(
    model=...,
    system_prompt="...",          # ← prefix on every run
)
```

**Door B: inside `invoke` — flexible, per call**

```python
agent.invoke({"messages": [
    SystemMessage("..."),          # ← 1st message, this call only
    HumanMessage("..."),
]})
```

> Rule: **A = same prompt every time · B = different prompt per question.**
> `model.invoke([...])` (no agent) → **only Door B exists** (a raw model has no `system_prompt=`).

## The 5-layer cheat sheet (copy-paste skeleton)

```python
# LAYER 1+2 · make the whole thing
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
import os

agent = create_agent(
    init_chat_model(
        model="gemini-3.1-flash-lite",          # WHICH model
        model_provider="google-genai",          # WHOSE model (google-genai | openai | groq | anthropic)
        api_key=os.getenv("GOOGLE_API_KEY"),    # its KEY from .env
    ),
    system_prompt="You are a science fiction writer...",   # LAYER 3 + Door A
)

question = HumanMessage(content="What is the capital of the Moon?")
response = agent.invoke({"messages": [question]})          # LAYER 2 step 3
print(response["messages"][-1].text)                       # LAYER 2 step 4

# LAYER 3, level ⑤  →  structured output
from pydantic import BaseModel
class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str
# ... create_agent(..., response_format=CapitalInfo)
result = response["structured_response"]   # an OBJECT: result.name works
```

## The ONE-SENTENCE master key

> *"Hire a worker, box your question, pass the box, read the last letter — and tighten
> the cage order: Basic, Persona, Examples, Labels, Schema. The prompt has two doors:
> inside the agent (fixed) or inside the call (per-question). Everything is one list:
> `messages`."*

---

# Tools & Invocation (Module 2 preview)

## What is a tool?
A tool = a normal Python function the model is allowed to *ask to run*. You define it; the agent runs it.

```python
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the internet."""
    return f"results for {query}"

search_web.invoke({"query": "capital of the moon"})   # manual test — works
search_web("capital of the moon")                     # shorter, identical
```

- `@tool` → turns your function into a `Tool`.
- `@tool("web")` → same, but renames the tool (the name the LLM sees).
- What the LLM sees per tool: **name** + **docstring** (description) + **type hints** (args schema).
- The structure: function decorator (`@tool`) → tool object → the agent decides when to run it.

## Do we invoke tools?
Usually **not by you** — the agent loop does it automatically:

```
model: "I want to call search_web(query='...')"   → tool_call (just TEXT, no data yet)
agent loop:  search_web.invoke({"query": "..."})  → real execution (HTTP/DB/API)
result → appended back into messages → model reads it → grounded final answer
```

## Why invoke the tool at all?
1. **The model can't run code.** It's a text engine — it outputs *intent* ("please call X"), not the actual call. Someone must execute it: that someone is the loop.
2. **Invocation = the adapter** between the model's JSON args and your Python function.
3. **Grounding is the point.** The final answer must be built on REAL tool output, not the model's guess ("I *would* check the weather" is useless).

## When YOU invoke a tool
1. **Testing** — `my_tool.invoke(...)` to verify it works before wiring it into an agent.
2. **Writing your own LangGraph loop** (Module 2/3) — you explicitly call `tool.invoke(tool_call["args"])`.

## The agent definition (one line)
```
decide → invoke → read → decide → ...   (repeat until no tool_call left)
```
**That loop IS an agent.** Give it no tools → answers directly. Give it tools → it can go fetch facts.

---

# How tools reach the model (the catalog)

## The key idea
Tools are NOT hidden code. Their **definitions are sent to the API on every call**, right next to the conversation — a "menu" the model reads before replying.

## What the model sees (serialized tool definition)

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current and forecast weather for a city. Use when the user asks about weather or temperature.",
    "parameters": {
      "type": "object",
      "properties": { "city": { "type": "string", "description": "The city name" } },
      "required": ["city"]
    }
  }
}
```

| JSON field | Comes from your code |
|---|---|
| `name` | function name (or `@tool("web")` string) |
| `description` | the **docstring** |
| `parameters` | the **type hints** |

> The model only knows a tool through this TEXT. Clear docstring = the model picks it correctly. Matching is **semantic** (question word "weather" ↔ description word "weather").

## The full tool flow step-by-step

```
AGENT CREATED      create_agent(model, tools=[get_weather])
                   └─ tools serialized into the catalog

1. USER ASKS       HumanMessage("What's the weather in Casablanca tomorrow?")
2. REQUEST BUILT   messages + tool catalog  →  sent to the API
3. MODEL READS MENU  picks get_weather (description fits the question)
4. MODEL EMITS tool_call   {name: get_weather, args: {"city": "Casablanca"}}
                            ← still just TEXT, no real data
5. LOOP NOTICES    a tool_call instead of plain text → looks up the tool by name
6. VALIDATION      args checked against the schema
7. EXECUTION       get_weather.invoke({"city": "Casablanca"}) runs the real code
8. RESULT APPENDED ToolMessage("+24°C, sunny tomorrow") added to messages
9. MODEL CALLED AGAIN  now sees the tool result
10. FINAL ANSWER   no more tool_call → loop stops → grounded answer ☀️
```

**One sentence:** *the model gets a menu of tools with every call, picks one by matching your words to the tool's description, asks for it (tool_call), the loop runs it for real, and feeds the real result back — that's how answers get grounded in facts.*

---

# LangChain × providers: string guessing vs. explicit pinning

## The problem in one line
> Bare `model="gemini-..."` crashes on Google, while it "just works" for OpenAI. NOT because of you — because LangChain **guesses** the provider from the string, and its guess for `gemini-*` lands on the wrong Google.

## A string has no pockets
A plain string can't carry a *provider* name or an *API key*. So `create_agent(model="...")` must **guess who your model's employer is**:

```
"gpt-5-nano"     → guess: openai        → reads OPENAI_API_KEY   → ✅ works
"claude-..."     → guess: anthropic     → reads ANTHROPIC_API_KEY → ✅ works
"groq:llama-..." → prefix pins groq     → reads GROQ_API_KEY     → ✅ works
"gemini-..."     → guess: VERTEX (!!)   → wants a GCP PROJECT    → 💥 GoogleAuthError
```

For a bare string to work, **two conditions must both hold**:
1. The guess is **right**, and
2. The provider auths with **just a key** (no project, no cloud account).

## Provider comparison

| Provider | You pass | Auth model | Bare string works? |
|---|---|---|---|
| OpenAI | `"gpt-5-nano"` | API key from env | ✅ guess right + key-only |
| Anthropic | `"claude-..."` | API key from env | ✅ guess right + key-only |
| Groq | `"groq:llama-..."` | API key from env | ✅ prefix needed + key-only |
| Google AI Studio | `"gemini-..."` bare | `GOOGLE_API_KEY` | ❌ guessed as VERTEX instead |
| Google Vertex | `"gemini-..."` bare | GCP project + service account | (would work, but you're NOT on Vertex) |

**Rule:** OpenAI/Anthropic/Groq = *"one key in env = done."* Google = *two different businesses that smell the same from a string*:
- **Vertex AI** (`google-vertexai`): Google Cloud, needs a project + cloud creds. LangChain's **default** for any `gemini-*` string.
- **AI Studio / Gemini API** (`google-genai`): free API key (`GOOGLE_API_KEY`). NOT what the guess picks.

> The crash: the `gemini-` string opens the **Vertex SDK**, which never even reads `GOOGLE_API_KEY` (it searches GCP project/credentials) → `GoogleAuthError: Unable to find your project`. Your key sat unused in `.env`.

## The word that fixes it: `google-genai`
There is exactly one string token that says *"I'm on AI Studio, the free-key one"* — **`google-genai`**.

```python
# v1 — explicit object (most readable)
model=init_chat_model(
    model="gemini-3.1-flash-lite",
    model_provider="google-genai",          # ← "AI Studio!"
    api_key=os.getenv("GOOGLE_API_KEY"),    # ← free key from .env
)

# v2 — provider pinned inside the string (same thing, terser)
model="google-genai:gemini-3.1-flash-lite"
```

Both make the guess unnecessary. The one rule that holds forever:

> **`model_provider` and `api_key` belong inside `init_chat_model(...)`. `create_agent(...)` and `tools=` only ever receive a READY model — or a string that already carries its `provider:` prefix.**

### 📌 TL;DR — using AI Studio (copy-paste ready)

**Issue:** bare `model="gemini-3.1-flash-lite"` → LangChain *guesses* **Vertex AI** → needs a GCP project → `GoogleAuthError` (`GOOGLE_API_KEY` never read).

**Fix 1 — explicit object:**
```python
from langchain.chat_models import init_chat_model
import os

model = init_chat_model(
    model="gemini-3.1-flash-lite",
    model_provider="google-genai",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
agent = create_agent(model=model, tools=[tool1])
```

**Fix 2 — string with `google-genai:` prefix:**
```python
agent = create_agent(
    model="google-genai:gemini-3.1-flash-lite",
    tools=[tool1],
)
```

---

# Mental model in one breath: `invoke` is a waiter

> **invoke is a waiter:** it takes your 1 message, shows it to the model, the model either (a) tells a story to you → done, or (b) orders tools → the waiter runs them, attaches a `ToolMessage` receipt via `tool_call_id`, shows everything to the model again, and only then returns — with the entire tape under `response["messages"]`. Every element is an **object**, `.tool_calls` is the model's order form, `content=[]` just means "ordering, not talking."

## The 4-message tape inside one `invoke`

```
[0] HumanMessage   ← YOUR Post-it (the ONLY message you wrote)
[1] AIMessage      ← model says NOTHING, only ASKS to use a tool   (content=[], tool_calls=[...])
[2] ToolMessage    ← the real result = tool.invoke(...) returned it   (+ tool_call_id link)
[3] AIMessage      ← model ANSWERS, grounded in that result         (content=[...], tool_calls=[])
```

## The hidden loop inside `invoke`

```
messages = [question]                       # 1 item
ai = model.invoke(messages)                 # round 1
messages.append(ai)                         # 2 items
while ai.tool_calls:                        # wants tools? → yes
    run each call → append ToolMessage(result, tool_call_id=call.id)
    ai = model.invoke(messages)             # round 2
    messages.append(ai)                     # 4 items → returned
return {"messages": messages}
```

## Key facts

- `response["messages"]` = dict envelope (key `"messages"`); messages **inside** are OBJECTS → access with `.` (`.content`, `.tool_calls`), NOT `[...]`.
- **Who made each message:** you made [0]; the **model** made [1] and [3]; the **loop** made [2].
- The one-for-one swap in every AI message:
  - `tool_calls NOT empty` + `content=[]` ⇒ model is **ordering**, not talking
  - `tool_calls=[]` + `content full` ⇒ model is **answering** → loop exits
- `tool_calls` = the model's order form: `[{name, args, id, type:'tool_call'}]`.
- `ToolMessage.tool_call_id` matches the AIMessage's `id` → links result to request.
- `.content` forms: plain `str` (Human/Tool) · `[]` (AI round 1) · `[{'type':'text','text':...}]` block-list (Gemini AI round 2) → final answer = `response["messages"][-1].content[0]["text"]` or `.text`.

---

# Tavily: ready-made tool vs. build your own

**Same backend, different packaging.** Both call Tavily's real web API and need your `TAVILY_API_KEY` (in `.env`). The API never disappears — only the boilerplate does.

| | `tavily.TavilyClient` (course, DIY) | `langchain_tavily.TavilySearch` (ready-made) |
|---|---|---|
| What it is | raw **HTTP client** for the Tavily API | raw client + **`@tool` wrapping done by the library** |
| Tool name | none — you write it | built-in (name + description + args schema) |
| Args schema | you write docstring + type hints | `{"query": "..."}` predefined |
| Goes in `tools=[...]` directly? | ❌ must `@tool`-wrap it first | ✅ drop-in |
| Output | raw dict `{query, results:[{url,title,content}...], answer}` | normalized string, LLM-friendly |
| Pros | full control + teaches you `@tool` | zero boilerplate, official |

**The two code shapes (identical result):**

```python
# DIY — you build the tool yourself (course cell 7)
from tavily import TavilyClient
from langchain.tools import tool
from typing import Dict, Any

tavily_client = TavilyClient()
@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

# Ready-made — the library already wrapped it as a tool
from langchain_tavily import TavilySearch
search = TavilySearch(max_results=5)          # reads TAVILY_API_KEY from env
```

**Notes:**
- Import path in installed v0.2.18: `from langchain_tavily import TavilySearch` (NOT `langchain_tavily.tool` — that submodule doesn't exist here).
- `langchain-tavily` is the **new** official integration, replacing the deprecated `langchain_community.tools.TavilySearchResults`. Newer ≠ key-free: it's still Tavily's API under the hood.
- Test standalone like any tool: `search.invoke({"query": "Who is the current mayor of San Francisco?"})`.
- Use in agent: `agent = create_agent(model=..., tools=[search], ...)` — same list slot as `tool1`.

---

# The 7-step agent recipe (always this order)

```
1. DETERMINE the model           → model = init_chat_model(model="...", model_provider="google-genai", api_key=...)
2. DETERMINE the tool (optional) → @tool def search_web(...)        (or TavilySearch(max_results=5))
3. DETERMINE the system prompt   → the character/instructions (optional)
4. CREATE the agent              → agent = create_agent(model=..., tools=[...], system_prompt=...)
      └─ the agent is the "worker + instruction sheet + toolbox" COMBINED
5. DETERMINE the user's message  → question = HumanMessage("...")
6. INVOKE the agent              → response = agent.invoke({"messages": [question]})
7. PRINT the response            → print(response["messages"][-1].text)
```

> ⚠️ `create_agent` is the **assembly** step — it must come AFTER 1–3 (you can't build the agent before its materials exist). `model_provider`/`api_key` only ever live in `init_chat_model` (step 1).

**Maps onto the 4 moves:** Worker = steps 1–4 (build the tooled-up model) · Post-it = step 5 (HumanMessage) · Box = step 6 (invoke) · Read = step 7 (print).

**What can be omitted:** [2] if no tools needed, [3] if no persona needed — the skeleton still holds.

---

# Memory & the stateless model (Module 1.3)

## The #1 principle
> **The model NEVER remembers anything.** It's a stateless function: `model.invoke(conversation_text) → reply_text`. Between calls nothing persists inside it. **"Memory" = the history being re-sent to the API on every turn.**

## Why an agent "forgets"
- The agent is a **graph** whose **state** = the `messages` list.
- Each `invoke({"messages": [...]})` starts from the messages *you* provide; when the run ends the state is returned — **and discarded unless saved**.
- Next `invoke` starts from scratch → model sees only your new message → honestly answers *"I don't know"*.

## What changes under the hood to make it remember
Two ingredients: a **checkpointer** (storage) + a **`thread_id`** (address).

```python
from langgraph.checkpoint.memory import InMemorySaver
agent = create_agent(model=model, checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "1"}}     # same key = same conversation
```

Every invoke now runs: **LOAD → APPEND → RUN → SAVE**

```
run 1: no saved state → [Human("Seán, green")] → model replies → SAVE under "1"
run 2: LOAD "1" → [Human, AI] → append new msg → model sees FULL transcript
       → remembers green ✅ → SAVE again (now longer)
```

| | No checkpointer | Checkpointer + thread_id |
|---|---|---|
| starting state of next run | fresh (`[Human(new)]`) | loaded (`[Human, AI, Human(new)]`) |
| what model sees | only your new message | the full previous transcript |
| state after run | discarded | saved under the thread key |
| verdict | forgets ❌ | remembers ✅ |

**The model changed in NOTHING** — only what text got prepended.

## What a thread is
- The checkpointer snapshots state after every graph step, keyed by `thread_id` (a **chain of checkpoints**, like git commits, each with `checkpoint_id`).
- Same `thread_id` = same drawer = remembers. Different `thread_id` = different drawer = separate conversation (each user = own thread).
- Lifecycle: *checkpointer = storage, `thread_id` = the address; address matters as much as storage — a key you never use again is a drawer you can never open.*

## The cost (and why memory isn't free)
- Every turn **re-sends the whole history** → tokens grow (real example: input_tokens 52 → 115 → 1894 once results arrived).
- Consequence: context-window limits → real apps trim/summarize old messages.
- `usage_metadata` per AI message is the receipt of that cost.

## "How to solve it" in general (the 4 memory strategies)
1. **Manually keep the list** (plain chat notebooks): you append every message and keep passing the list — works, full control.
2. **Checkpointer (the agent way):** graph state auto-saved under `thread_id` — automatic history persistence (Module 1.3).
3. **Short-term vs long-term:** short-term = message history within a thread; long-term = a memory store (vector DB of past facts) the agent retrieves when relevant.
4. **Compression:** summarize/trim old messages when the thread grows — trade fidelity for context space.

> One-liner: *the notebook's chat list, the agent's checkpointer, and a vector memory store are all the same trick — get the relevant history back IN FRONT of the model before it answers.*

## Multimodal: how images & audio become tokens

> One-liner: *base64 is the TAXI that carries the bytes inside JSON — the encoders are the METAMORPHOSIS that turns pixels/sound waves into the LLM's own language: **tokens**.*

### Two separate problems
1. **Transport** — you must GET the bytes to the model's server. JSON is text-only, so binary gets re-encoded as text: **base64** (6 bits → 1 ASCII char, file ~+33% bigger). The client sends `{"type": "image", "base64": "...", "mime_type": "image/png"}`; the server decodes it back to bytes.
2. **Understanding** — the transformer only reads **token embeddings** (built from text). Raw pixels / raw audio samples mean nothing to it. Something must *convert* them.

### The pipeline (image)
```
PNG bytes
   │  server decodes base64 → pixel matrix (H×W×3)
   ▼
VISION ENCODER  (CLIP/ViT — slices the image into 16×16 patches)
   │  each patch → a vector capturing edges, shapes, textures
   ▼  (hundreds of vectors: too many, too much pixel-level detail)
PROJECTOR  (a couple of neural-net layers)
   │  maps those vectors → IMAGE TOKENS
   ▼  (same shape as word-token embeddings → SAME latent space)
TRANSFORMER
   │  attention mixes image tokens with text tokens
   ▼
answer tokens → your `response['messages'][-1].content`
```

### The pipeline (audio)
```
WAV bytes
   │  server decodes base64 → samples
   ▼
AUDIO SEGMENTER  (chunks, ~10–30 s)
   │
   ▼
AUDIO EMBEDDER  (waveform → spectral features → vectors)
   ▼
audio embeddings → AUDIO TOKENS  (same token space as text)
   ▼
TRANSFORMER attention  (fuses text + audio)
   ▼
answer tokens
```

### Key facts
- **"To the LLM, vision/audio is just more tokens in the prompt."** Same attention, same next-token prediction — it never "sees" or "hears".
- **Why base64 at all?** Because the API contract is JSON, and JSON has no binary type (same reason email uses MIME/base64 attachments). The taxi, not the brain.
- **Content blocks = a tagged union:** `{type: text|image|audio, ...payload}` — the `type` tells the parser which encoder to route the payload through. Answers come back in the *same* block shape (full symmetry: you send blocks, it answers with blocks).
- **Context cost:** one small image ≈ hundreds/thousands of tokens (vs ~4 chars/token for text). Multimodal eats context.
- **High-res:** modern VLMs *tile* large images (anyres) so small text isn't smeared — but tokens multiply again.

### Verified on Gemini (module 1.4, free `gemini-3.1-flash-lite`)
- 64×64 red PNG → "crimson/deep red" ✅
- user's real `vangogh.png` (1.5 MB) → described an invented capital city ✅
- 440 Hz sine WAV → "the sound of a beep" ✅

### Why the RECORDING cell in 1.4 fails (`PortAudioError: Error querying device -1`)
- `sounddevice` records on the machine running the Python **kernel**. The kernel runs in **WSL-Linux**, and WSL exposes **no mic input by default** (WSLg outputs speakers, not mic-in) → OS reports "no default input device" → PortAudio's `-1` = "default" doesn't exist.
- **Not** a code/model problem. Fix: record on Windows (Voice Recorder / Win+Shift+S → `.wav`), then **browser-upload** it via a `FileUpload` widget — exactly like the image cell.

## MCP: the USB-C of AI integrations

> One-liner: *MCP solves the **integration jungle** — one wire for every AI app ↔ every tool.*

### Before MCP
Every app-model pair needed **custom glue** (SDK setup, auth, call formatting, error handling):

```
app A (Gemini) ──custom code── Slack
app A (Gemini) ──custom code── GitHub
app B (Claude) ──custom code── Slack   ← same Slack, rewritten again
```

**N apps × M tools = N×M hand-written connectors** — duplicated plumbing, nothing portable.

### After MCP
One open standard (JSON-RPC 2.0, Anthropic Nov 2024) for apps ↔ data/tools:

```
app A ─┐
app B ─┼─ MCP client ── MCP server ── Slack / GitHub / DB / Drive / anything
app C ─┘               (one implementation serves every app)
```

### The 3 hats it standardizes
| Hat | Role (chef version) | Server decorator |
|---|---|---|
| **Tools** | hands — functions the model can *call* (`web_search`) | `@mcp.tool()` |
| **Resources** | files — data the app can *pull* by URI (recipe index) | `@mcp.resource("uri://...")` |
| **Prompts** | script — reusable system-prompt templates (chef persona) | `@mcp.prompt()` |

**Analogy:** before MCP every device had its own custom port; MCP is **USB-C** — a tool maker builds *one* server, every MCP app uses it; an app ships *one* client, every MCP server plugs in. (Module 2: server file `resources/2.1_mcp_server.py` + client `MultiServerMCPClient`, async `await client.get_tools()`, `model="gpt-5-nano"` → swap to Gemini.)

### Mental-model flow (in order)
> **Build → Decorate → Serve → Connect → Collect → Assemble → Ask** *(server first, then client)*

1. **Build** — `mcp = FastMCP("name")` (kitchen opens)
2. **Decorate** — hang the 3 hats: `@mcp.tool()` / `@mcp.resource("uri")` / `@mcp.prompt()`
3. **Serve** — `mcp.run(transport="stdio")` under `__main__`
4. **Connect** — `MultiServerMCPClient({"name": {transport, command, args}})`
5. **Collect** — `await get_tools()` + `get_resources(name)` + `get_prompt(name, "prompt")` → unwrap `[0].content`
6. **Assemble** — `create_agent(model, tools, system_prompt)`
7. **Ask** — `await agent.ainvoke({"messages": [...]})`

**Traps:** client is async (`await` everywhere); server secrets never leave the server; `model="gpt-5-nano"` → Gemini swap at step 6.

### MCP client ↔ server ↔ tool (mermaid)
```mermaid
sequenceDiagram
    autonumber
    participant AG as Agent (llm + create_agent)
    participant MC as MCP Client (MultiServerMCPClient)
    participant TP as Transport (stdio pipe or HTTP)
    participant SE as MCP Server (FastMCP "local_server")
    participant TO as Tool (search_web → Tavily)

    AG->>MC: get_tools()
    MC->>TP: JSON-RPC "tools/list"
    TP->>SE: tools/list
    SE-->>TP: registered @mcp.tool()s
    TP-->>MC: JSON: tool schemas
    MC-->>AG: LangChain tools

    AG->>AG: create_agent(model, tools, system_prompt)

    AG->>MC: tool call search_web("chicken rice")
    MC->>TP: JSON-RPC "tools/call" {name, arguments}
    TP->>SE: tools/call
    SE->>TO: query
    TO-->>SE: results (Tavily)
    SE-->>TP: JSON result
    TP-->>MC: result
    MC-->>AG: ToolMessage
    AG->>AG: llm reads results → final answer
```

### The build pipeline (server → client → agent)
```mermaid
flowchart LR
    S["mcp = FastMCP('mcp_server')"]
    T["@mcp.tool() search_web"]
    R["@mcp.resource('uri://...')"]
    P["@mcp.prompt() prompt"]
    RUN["mcp.run transport='stdio'"]
    C["MultiServerMCPClient({ ... spawn ... })"]
    GT["await get_tools()"]
    GR["await get_resources(name)"]
    GP["await get_prompt(name,'prompt')"]
    SP["system_prompt = prompt[0].content"]
    CA["create_agent(model, tools, system_prompt)"]
    AI["await agent.ainvoke({...})"]

    S --> T
    S --> R
    S --> P
    T --> RUN
    R --> RUN
    P --> RUN
    RUN -- "stdio pipe (JSON-RPC)" --> C
    C --> GT
    C --> GR
    C --> GP
    GP --> SP
    GT --> CA
    SP --> CA
    CA --> AI

    classDef s fill:#16324f,stroke:#4c8bf5,color:#e8f1ff;
    class S,T,R,P,RUN s
    classDef c fill:#123c33,stroke:#2ecc9b,color:#e6fff7;
    class C,GT,GR,GP,SP c
    classDef a fill:#4a1630,stroke:#f56c9b,color:#ffe3ef;
    class CA,AI a
```

### Where the tools go (server → client → agent)
> Tools flow **one way**: the server publishes them, the client fetches & converts them, the agent holds and uses them.

```
MCP SERVER                       CLIENT                          AGENT
(publishes tools)              (fetches & converts)          (holds the tools)
─────────────────              ───────────────────             ──────────────
@mcp.tool() search_web   ──►   await client.get_tools()  ──►   tools = [search_web, ...]
@mcp.tool() get_stocks   ──►        "hand me your hats"  ──►   create_agent(model, tools=tools)

         tools flow ONE WAY:  server ──► client ──► agent
```

- The **server owns** the tools; the **agent borrows** them (via the client + stdio pipe).
- When the model calls `search_web`, the client relays it over the pipe and the **server executes** it (with its own secrets) before returning the result.
- And it's a **choice**: the model calls a tool only when it needs info it doesn't have AND the tool's description matches (e.g. "2+2" → no call; "tell me about langchain-mcp-adapters" → calls `search_web`).

## Multi-agent architecture (tiered) — 2.3 pattern

### Schema (mermaid)

```mermaid
flowchart TD
    subgraph T3["Tier 3 · MAIN / ORCHESTRATOR (router)"]
        MAIN["main_agent = create_agent(model, tools=[call_subagent_1, call_subagent_2], system_prompt=...)"]
        ROUTER{{"routes by NAME + DESCRIPTION<br/>(never sees leaf tools)"}}
        MAIN --- ROUTER
    end

    subgraph T2["Tier 2 · SUBAGENT AS TOOL (bridge)"]
        W1["@tool call_subagent_1(x)<br/>→ subagent_1.invoke(messages)<br/>→ returns response.messages[-1].content"]
        W2["@tool call_subagent_2(x)<br/>→ subagent_2.invoke(messages)<br/>→ returns response.messages[-1].content"]
    end

    subgraph T1["Tier 1 · SUBAGENTS (workers, one specialty each)"]
        S1["subagent_1 = create_agent(model, tools=[square_root])"]
        S2["subagent_2 = create_agent(model, tools=[square])"]
    end

    subgraph T0["Tier 0 · LEAF TOOLS (real abilities)"]
        L1["@tool square_root(x)<br/>return x ** 0.5"]
        L2["@tool square(x)<br/>return x ** 2"]
    end

    Q(["Question:<br/>'sqrt of 456?'"]) --> MAIN
    ROUTER -->|"matches desc 'square root'"| W1
    ROUTER -->|"matches desc 'square'"| W2
    W1 --> S1
    W2 --> S2
    S1 --> L1
    S2 --> L2
    L1 -.->|"number comes<br/>BACK UP the tiers"| W1
    W1 -.-> MAIN
    MAIN -.->|"final answer"| A(["Answer to user"])

    classDef t3 fill:#4a1630,stroke:#f56c9b,color:#ffe3ef;
    classDef t2 fill:#5c3a00,stroke:#f0a92b,color:#fff3d9;
    classDef t1 fill:#123c33,stroke:#2ecc9b,color:#e6fff7;
    classDef t0 fill:#16324f,stroke:#4c8bf5,color:#e8f1ff;
    class MAIN,ROUTER,MAIN t3;
    class W1,W2 t2;
    class S1,S2 t1;
    class L1,L2 t0;
```

### Rules of the stack

1. **An agent is NOT a tool** — wrap each subagent in a `@tool` that does `subagent_N.invoke({...})` and returns `response["messages"][-1].content` (Tier 2 is the bridge).
2. **Routing by description** — Tier 3 only sees `call_subagent_1` ("…square root…") / `call_subagent_2` ("…square…"); the **docstrings are the menu**.
3. **Each tier owns its loop** — Tier 1 runs its own `model → leaf tool → model` cycle internally.
4. **One leaf-tool = one subagent** — separation of concern: main gets *capability*, leaf gets *implementation*.

### Data flow

```
Question: "What is the square root of 456?"
  1. TIER 3  main reads catalogue → desc matches "square root"
  2.        → calls call_subagent_1(456)          (main does NOT do math)
  3. TIER 1  subagent_1 runs its loop → square_root(456)         ← TIER 0
  4.        → returns result up to the wrapper
  5. TIER 2  wrapper returns response["messages"][-1].content → main
  6. TIER 3  main formats the final answer for the user
```

## The Ultimate LangChain Canvas (reuse for ANY agent you build)

### Why LangChain (vs raw API calls)
- **One uniform API** over every provider — swap OpenAI ⇄ Gemini in one line (`init_chat_model`).
- **One normalizer for everything external** — `@tool`, MCP `get_tools()`, retrievers, DBs all become the same shape: `name + description + JSON schema`.
- **The loop is handled** for you: decide-to-call → execute → feed ToolMessage back → repeat, until final answer.
- **Memory, state, context, streaming, tracing (LangSmith)** all plug into the same objects.
- **It sits on LangGraph** — start simple (`create_agent`), drop to graph-level control when needed.

### LangChain vs LangGraph
| | **LangChain** (`create_agent`) | **LangGraph** (raw graphs) |
|---|---|---|
| abstraction | high-level: **blocks** | low-level: **nodes + edges** |
| control flow | the loop is prebuilt & opinionated | YOU design the flow (branch, cycle, checkpoint mid-run) |
| good for | 80% of agents: tools + loop + memory | custom flows: HITL approval, cycles, parallel branches, state machines |
| mental model | "give it a brain, hands, prompt" | "draw a program as a state machine, then run it" |
| relationship | **`create_agent` is a LangGraph graph under the hood** | the engine LangChain's agents are built on |

Rule of thumb: **build with LangChain first; drop to LangGraph only when you need to draw your own arrows.**

### The 6 LAYERS (the canvas)

```
LAYER 4 · THE LOOP (who talks, in what order)      ← LangGraph drives it
         model node ⇄ tool node  (repeat until no tool_calls)
───────────────────────────────────────────────────────────────
LAYER 3 · THE AGENT (how it's combined)
         create_agent(model, tools, system_prompt, state_schema, checkpointer, middleware)
───────────────────────────────────────────────────────────────
LAYER 2 · THE BRAIN (any provider)
         model = init_chat_model(...) → model.invoke / ainvoke
───────────────────────────────────────────────────────────────
LAYER 1 · THE HANDS (what you expose)
         @tool fns · MCP get_tools() · retriever · SQL/web
───────────────────────────────────────────────────────────────
LAYER 0 · THE FUEL (inputs + memory)
         messages list · State (suitcase) · Context (briefing) · checkpointer(thread_id)
```

### The universal 5-step build (EVERY LangChain agent)
1. **Hands** — `@tool` / `client.get_tools()` / retriever → the catalog.
2. **Memory shape** — `state_schema=` and/or `context_schema=` + `checkpointer=` (optional).
3. **Brain** — `model = init_chat_model("gemini-3.1-flash-lite", ...)` once.
4. **Agent** — `create_agent(model=model, tools=..., system_prompt=..., state_schema=..., checkpointer=...)`.
5. **Run** — `resp = agent.invoke({"messages":[HumanMessage(...)]}, config)` → read `resp["messages"][-1].content`.

### The universal FLOW (mental flux of every run)
```
user text → {"messages":[HumanMessage]} → create_agent's LOOP:
  model sees messages + tool catalog
      → no tool needed?      answer → DONE  (read response["messages"][-1].content)
      → tool needed?         CALL tool (local fn / MCP / DB / web)
                             ToolMessage back into messages
                             loop again with tool result … repeat
  checkpointer snapshots state after each step under thread_id
  state/context feed the loop extra data (mutable vs read-only)
```

### Mermaid — the flow

```mermaid
flowchart TD
    U["User text"] --> M0["INPUT<br/>{messages:[HumanMessage]}<br/>+ state/context"]
    M0 --> CA

    subgraph AGENT["LangChain create_agent  (builds a LangGraph graph)"]
        CA["create_agent(model, tools, system_prompt,<br/>state_schema, checkpointer)"]
        CA --> LOOP{"MODEL NODE<br/>reads messages + tool catalog"}
        LOOP -- "emits tool_calls" --> TOOLS["TOOL NODE<br/>executes catalog"]
        LOOP -- "no tool_calls" --> FIN["final answer"]
        TOOLS -- "ToolMessage → messages" --> LOOP
    end

    subgraph HANDS["LAYER 1 · HANDS"]
        T1["@tool fns"]
        T2["MCP server<br/>(get_tools)"]
        T3["retriever / DB / web"]
    end

    subgraph FUEL["LAYER 0 · FUEL"]
        ST["State (mutable, persisted)"]
        CT["Context (read-only, per-call)"]
        CP["checkpointer + thread_id"]
    end

    subgraph BRAIN["LAYER 2 · BRAIN"]
        B["init_chat_model<br/>Gemini / OpenAI / ..."]
    end

    TOOLS --> HANDS
    HANDS --> TOOLS
    CA -.reads.-> BRAIN
    FUEL -.reads/writes.-> CA
    FIN --> OUT["resp['messages'][-1].content"]

    classDef lc fill:#16324f,stroke:#4c8bf5,color:#e8f1ff;
    classDef lg fill:#4a1630,stroke:#f56c9b,color:#ffe3ef;
    classDef h fill:#123c33,stroke:#2ecc9b,color:#e6fff7;
    class CA,LOOP,TOOLS,FIN,OUT lc;
    class ST,CT,CP h;
    class B h;
```

### The UNIVERSAL AGENT SKELETON (apply to ANY agent)

| # | Step (generic) | Wedding example |
|---|---|---|
| **0** | **Setup** — load `.env`, imports | `load_dotenv()` |
| **1** | **Create the model** (one brain, reuse everywhere) | `model = init_chat_model("gemini-3.1-flash-lite", ...)` |
| **2** | **Shape State** *(skip if single-agent / no cross-turn data)* | `class WeddingState(AgentState): origin, destination, guest_count, genre` |
| **3** | **Create the tools (Tier 0)** — local `@tool` fns + **MCP only if data belongs to someone else** | `@tool web_search` (Tavily) · `@tool query_playlist_db` (SQLite) · `tools = await client.get_tools()` (Kiwi MCP) |
| **4** | **Create subagents (Tier 1)** — ONE per specialty, own tools + own rules | `travel_agent` · `venue_agent` · `playlist_agent` |
| **5** | **Wrap subagents as tools (Tier 2)** — `@tool def call_X(runtime): subagent_X.invoke(...) → return response["messages"][-1].content` | `search_flights`, `search_venues`, `suggest_playlist` (+ `update_state` = `Command(update)`) |
| **6** | **Create the main agent (Tier 3)** — `create_agent(model, tools=[wrappers], state_schema=..., system_prompt=workflow)` | `coordinator = create_agent(model, tools=[...], state_schema=WeddingState, system_prompt=...)` |
| **7** | **Run it** — `invoke` (sync) or `await ainvoke` (async/MCP) | `await coordinator.ainvoke({"messages":[HumanMessage(...)]}, config={"recursion_limit":40})` |
| **8** | **Read the answer** — `response["messages"][-1].content` | `print(response["messages"][-1].content)` |
| **9** | **Verify** — check flow, prompts, edge cases | state collected first? specialists answered? |

### Decision key (when to CUT steps)
- **Single agent, no memory across calls** → skip 2, 4, 5, 6 ⇒ just **1 → 3 → create_agent → 7**.
- **No third-party live service** → skip the MCP part of step 3.
- **Read-only inputs, no persistence** → use `context_schema=` instead of `state_schema=`.
- **Parallel flow / human-in-the-loop / custom branching** → at step 6 drop into **raw LangGraph** (draw your own nodes/edges).

### The one-liner that explains every agent
**"A brain (model) with hands (tools) following rules (system_prompt), carrying a suitcase (state/context), looping model⇄tool until it has your answer."**

### The AI Agent at a glance (what the 4 pillars do)

```
                    AI AGENT
                       │
        ┌──────────────┼───────────────┐
        │              │               │
      STATE          CONTEXT       MIDDLEWARE
        │              │               │
   remembers       gives useful    controls/
   workflow        information     modifies flow
                                        │
                                        │
                                       MCP
                                        │
                              connects external
                                  capabilities
```

- **State** → remembers workflow (mutable, persisted, `Command(update)`, checkpointer).
- **Context** → gives useful info (read-only, per-call, `runtime.context`).
- **Middleware** → controls/modifies flow (wrap model & tool calls, guards, dynamic prompts).
- **MCP** → connects external capabilities (tools/resources/prompts from remote or local servers).

### The LangChain stack (what's inside, top → bottom)

```
                    LangChain
                       │
        ┌──────────────┼─────────────┐
        ↓              ↓             ↓
      Agents         Tools         Middleware
        │
        ↓
      State
        │
        ↓
      Context
        │
        ↓
    external systems
        │
        ↓
       MCP
```

Reading it as layers: **Agents** (driven by **Tools** + **Middleware**) sit on **State** (workflow memory) → **Context** (per-call info) → reach **external systems** through **MCP**.

### FastMCP vs MultiServerMCPClient

| | **FastMCP** | **MultiServerMCPClient** |
|---|---|---|
| role | build a server (expose your tools) | connect to servers (fetch tools) |
| side | the **chef/kitchen** | the **waiter** |
| who uses it | Kiwi (their side) & you in 2.1 server file | you (your notebook/agent) |

Same MCP protocol, opposite sides: FastMCP **serves** tools (`@mcp.tool()`), MultiServerMCPClient **consumes** them (`get_tools()` → normal LangChain tools).

### SQL connection URIs (cheat-sheet)

Put these in `SQLDatabase.from_uri(...)`:

| Target | URI |
|---|---|
| SQLite file | `sqlite:///path/to/file.db` |
| Postgres | `postgresql://user:password@host:5432/dbname` |
| MySQL | `mysql+pymysql://user:password@host/dbname` |
| SQLite in memory (test) | `sqlite:///:memory:` |

Recipe to create your own SQLite from a CSV:

```python
import sqlite3, pandas as pd
df = pd.read_csv("mes_donnees.csv")
con = sqlite3.connect("resources/MonDB.db")      # creates the file
df.to_sql("ma_table", con, if_exists="replace", index=False)
con.close()
```

## Managing long conversations (Middleware · Agent · Checkpointer · Store)

> *The full home of an agent request — out in front: **middleware** checks the message; the **agent** runs the loop; the **checkpointer** snapshots the thread; the **store** remembers across threads.*

```
                    USER
                     │
                     ▼
              ┌─────────────┐
              │ Middleware  │
              │             │
              │ Trim?       │
              │ Summarize?  │
              │ Other logic?│
              └──────┬──────┘
                     │
                     ▼
                  AGENT
                     │
             ┌───────┴───────┐
             ▼               ▼
            LLM             TOOLS
             │
             ▼
          RESPONSE


CHECKPOINTER
     │
     └── saves conversation state
          by thread_id


STORE
     │
     └── saves long-term user information
          across threads
```

### Middleware — the guard corridor (before the agent, and around each call)

- The **first thing** a user message hits. Cross-cutting logic that applies to *every* request, without touching the agent itself.
- Hooks: `before_agent` / `after_agent`, `before_model` / `after_model`, `wrap_model_call`, `wrap_tool_call`, `transformers` (`AgentMiddleware(transformers=[change_state])`).
- **Trim? Summarize? Other logic?** all live here.

| Decision | Why | API in this venv |
|---|---|---|
| **Trim** (delete old) | free up tokens cheaply, lossy | `AgentMiddleware(transformers=[def trim(state, runtime): ...])` or `trim_messages(..., max_tokens=...)` |
| **Summarize** (compress old) | keep meaning, lose detail | `SummarizationMiddleware(model=model, trigger=("tokens", 40000), keep=("messages", 20))` |
| Other | logging, guards, dynamic prompts | `AgentMiddleware(awrap_tool_call=..., dynamic_prompt=...)` |

### Agent — the loop (brain + hands)

- Receives the (possibly trimmed/summarized) message; **loops LLM ⇄ TOOLS** until it has a final answer.
- The whole conversation lives in `state["messages"]` → that list is what middleware trims/summarizes.
- `create_agent(model=..., tools=..., middleware=[...], checkpointer=..., store=...)`.

### LLM + TOOLS — the two hands of the loop

- **LLM** = the brain, reads the message history (the walking tape). **TOOLS** = the abilities the model can call in a loop (MCP, Tavily, SQL, sub-agents…).
- The model decides *when* to call a tool; the tool output comes back as a message into the same history.

### Checkpointer — memory WITHIN a conversation

- Saves a **snapshot of the whole state** (messages + custom fields) **by `thread_id`** after every step.
- Same `thread_id` = the conversation **resumes** (any number of turns, across calls/restarts). Different id = separate conversation.
- `create_agent(..., checkpointer=InMemorySaver())` → `config={"configurable": {"thread_id": "1"}}`.
- `InMemorySaver` = RAM only (learning). `SqliteSaver` / Postgres = survives restarts (not installed here).

### Store — memory ACROSS conversations

- Long-term user facts that outlive any single thread (e.g., "the wedding is in Paris, 100 guests, jazz").
- The agent **saves** via a tool during a conversation, **retrieves** in a later one ("semantic memory").
- `create_agent(..., store=InMemoryStore())`; write/query with namespaces `(namespace, key)`.

### How it all combines (module 3 pattern)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model, tools=tools,
    middleware=[SummarizationMiddleware(model=model, trigger=("tokens", 40000))],
    checkpointer=InMemorySaver(),          # within-thread memory
    store=InMemoryStore(),                 # across-thread memory
)
agent.invoke({"messages": [("user", "hi")]},
             config={"configurable": {"thread_id": "1"}})
```

**One-liner:** *Trim deletes, Summarize compresses, Checkpointer resumes the same thread, Store remembers across threads.*

Then: `db = SQLDatabase.from_uri("sqlite:///resources/MonDB.db")` → test with `db.run(...)` → wrap in a `@tool` with `try/except` → in the system prompt: *"Discover the schema first, then query."*

## The wedding agent walkthrough (a real question, step by step)

> *One sentence in → **state first**, then **3 parallel experts**, then **one coordinated answer**. Every tool result is just another message on the same tape.*

**Question:** *"I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre"*

### TURN 1 — the state gate (1 tool call)

The model reads the sentence, sees all 4 facts are present, so it calls:

```
update_state(origin="London", destination="Paris", guest_count="100", genre="jazz")
```

- A **wrapper tool** — no sub-agent, no leaf tool.
- Writes into state via `Command(update=...)` + returns `ToolMessage("Successfully updated state")`.
- **Must run first** so the other 3 tools can read `runtime.state` when they fire.
- It only reads the model's hidden state — nothing comes from the outside yet.

### TURN 2 — the delegation (3 tool calls **in parallel**)

State is filled → the model fires 3 `tool_calls` in **one** message:

| Wrapper tool | Delegates to | Leaf tools (real abilities) |
|---|---|---|
| `search_flights` (async) | `travel_agent.ainvoke(...)` | Kiwi **MCP** tools: `search-flight` (`https://mcp.kiwi.com`) → real flights LGW→CDG €58 |
| `search_venues` | `venue_agent.invoke(...)` | **Tavily** `web_search` (max 12 searches) → venues (€5,800 barn, château, Shangri-La) |
| `suggest_playlist` | `playlist_agent.invoke(...)` | `query_playlist_db` → `db.run("SELECT ... FROM Track ...")` on Chinook → 130 jazz tracks |

- Each sub-agent is **its own `create_agent` loop**: model ⇄ its own tools until it answers.
- The wrapper returns `response["messages"][-1].content` — a **str** that becomes a `ToolMessage` in the coordinator's state.

### TURN 3 — the coordination

All 3 results are now messages. The coordinator calls **no more tools** (`finish_reason: STOP`) and writes the final plan: flights + venues + playlist, one polished answer.

### Where tools "intervene" (layer 0 → 3 — isolation is the point)

```
0  coordinator sees ONLY: update_state · search_flights · search_venues · suggest_playlist
   ├─ 1  travel_agent   sees ONLY Kiwi tools
   ├─ 1  venue_agent    sees ONLY the web_search tool
   └─ 1  playlist_agent sees ONLY query_playlist_db
```

- The coordinator never touches leaf tools; sub-agents never touch each other's.
- `recursion_limit=40` exists so the nesting (coordinator → sub-agent → leaf) doesn't hit the step ceiling.

### Why the SQL path broke (and the fix)

```python
SQLDatabase.from_uri("sqlite:///resources/Chinook.db")   # RELATIVE to kernel cwd
```

- SQLite resolves a relative path from the **kernel's working directory** — not the notebook's folder.
- JupyterLab: cwd = notebook folder → works. **VS Code: cwd = workspace root** → `resources/` missing → `OperationalError: unable to open database file`.
- **Rule:** *relative paths in a notebook resolve against the kernel's cwd* — when a path bugs you, print `os.getcwd()`.

Fix (path-anchored, works from any cwd):

```python
from pathlib import Path
from langchain_community.utilities import SQLDatabase

_db = next((p for p in [
    Path("/absolute/path/resources/Chinook.db"),
    Path.cwd() / "notebooks" / "module-2" / "resources" / "Chinook.db",
    Path.cwd() / "resources" / "Chinook.db",
] if p.exists()), None)
db = SQLDatabase.from_uri(f"sqlite:///{_db}")
```

## Python return types for chatbots (dict vs list reference)

> *A `dict` stores **one** object with named fields (keys); a `list` stores **many** values in order — often a `list` of `dict`s.*

| Case | Usually returns | Example output | How to access it |
|---|---|---|---|
| One object/result | `dict` | `{"article": "505", "similarity": 0.91, "source": "code-penal.pdf"}` | `result["article"]` |
| Multiple objects/results | `list[dict]` | `[{"article": "505", "score": 0.91}, {"article": "506", "score": 0.85}]` | `results[0]["article"]` |
| Multiple messages | `list[dict]` | `[{"role": "user", "content": "ما عقوبة السرقة؟"}, {"role": "assistant", "content": "يعاقب القانون..."}]` | `messages[0]["content"]` |
| One user's information | `dict` | `{"id": 1, "name": "Imane", "role": "student", "city": "Casablanca"}` | `user["name"]` |
| Search results | `list[dict]` | `[{"title": "مدونة الأسرة", "url": "https://example.com", "snippet": "الفصل 171..."}]` | `results[0]["url"]` |
| One document | `dict` or object | `{"id": 12, "filename": "code-famille.pdf", "page": 17, "text": "مادة 171..."}` | `document["filename"]` |
| Multiple documents | `list[dict]` | `[{"id": 1, "title": "القانون الجنائي"}, {"id": 2, "title": "مدونة الأسرة"}]` | `documents[1]["title"]` |
| One tool result with fields | `dict` | `{"city": "Casablanca", "temperature": 27, "weather": "sunny"}` | `tool_result["temperature"]` |
| Multiple tool results | `list` | `[{"tool": "weather", "result": {"temperature": 27}}, {"tool": "time", "result": {"time": "14:30"}}]` | `results[0]["result"]` |
| Agent state | `dict` | `{"task": "حضانة الأطفال", "status": "completed", "answer": "..."}` | `state["status"]` |
| Messages inside LangGraph state | `list[dict]` | `{"messages": [{"role": "user", "content": "ما هي الحضانة؟"}, {"role": "assistant", "content": "..."}]}` | `state["messages"][-1]` |
| Complete LangGraph state | `dict` | `{"messages": [...], "sources": [...], "answer": "...", "retry_count": 0}` | `state["answer"]` |
| No result found | `None` | `search("unknown topic") → None` | `if result is None:` |
| One selected option | `str` | `"model": "gemini-3.1-flash-lite"` | `result["model"]` |
| Multiple selected options | `list[str]` | `["القانون الجنائي", "مدونة الأسرة", "مدونة الشغل"]` | `options[0]` |
| Number of results | `int` | `len(results) → 5` | `print(count)` |
| Yes/no result | `bool` | `document_exists → True` | `if document_exists:` |
| One database row | `dict` | `{"id": 1, "law": "القانون الجنائي", "article": "505"}` | `row["article"]` |
| Multiple database rows | `list[dict]` | `[{"article": "505"}, {"article": "506"}]` | `rows[0]["article"]` |
| API response, even with one result | `dict` | `{"answer": "النص...", "sources": [{"article": "505"}]}` | `response["answer"]` |
| API error response | `dict` | `{"error": "Model unavailable", "status": 503}` | `response["error"]` |

The primary structural difference is the return type: a `dict` stores one object with named fields using keys, while a `list` stores multiple values in order, often containing multiple dictionaries.

## Python's useful built-in errors (the common families)

> *You don't need to memorize every Python exception — learn the common families.*

| Error | Meaning |
|---|---|
| `SyntaxError` | Python code syntax is invalid |
| `NameError` | Variable/name doesn't exist |
| `TypeError` | Wrong type/operation |
| `ValueError` | Correct type, inappropriate value |
| `KeyError` | Dictionary key doesn't exist |
| `IndexError` | List index doesn't exist |
| `AttributeError` | Object doesn't have that attribute |
| `FileNotFoundError` | File doesn't exist |
| `PermissionError` | You don't have permission |
| `ZeroDivisionError` | Division by zero |
| `ImportError` | Import problem |
| `ModuleNotFoundError` | Module/package can't be found |
| `JSONDecodeError` | Invalid JSON |
| `TimeoutError` | Operation timed out |
| `ConnectionError` | Connection failed |
| `FileExistsError` | File/directory already exists |

**Families to recognize in your code:** name/attribute/type (`NameError`, `AttributeError`, `TypeError`) → the classic `.xx` or missing-import mistakes; container access (`KeyError`, `IndexError`) → empty/missing results from your tools (`if result is None:` or `if "error" in result`); I/O & network (`FileNotFoundError`, `ConnectionError`, `TimeoutError`) → user files and API calls; parsing (`JSONDecodeError`, `ValueError`) → trusting external input.

## The full request flow (client → API → logic → DB / LLM / API) + where errors appear

> *Your chatbot's life as one round-trip: HTTP in → Python model → business logic → outside worlds (DB / LLM / APIs) → serialize → HTTP out.*

```
                CLIENT
                  │
                  │ HTTP
                  ▼
             ┌─────────┐
             │  API    │
             │FastAPI  │
             │ Flask   │
             └────┬────┘
                  │
             parse JSON
                  │
                  ▼
          Python data/model
                  │
                  ▼
             business logic
                  │
       ┌──────────┼──────────┐
       ↓          ↓          ↓
      DB         LLM        API
       │          │          │
       └──────────┼──────────┘
                  ↓
              result
                  │
                  ▼
          serialize to JSON
                  │
                  ▼
                CLIENT
```

| Stage of the flow | What happens there | Typical error there |
|---|---|---|
| **HTTP in** | client sends JSON to FastAPI/Flask route | `ConnectionError`, `TimeoutError` (client side) |
| **parse JSON** | body → Python structures | `JSONDecodeError` (invalid body), `ValueError` |
| **data/model** | JSON → your own objects/dicts | `KeyError` (missing field), `TypeError` |
| **business logic** | your rules, validations | `ValueError` (right type, bad value) |
| **DB** | query/insert | `FileNotFoundError` (SQLite file), `KeyError`/`IndexError` on rows |
| **LLM** | model call | `TimeoutError`, `ConnectionError`, invalid key → `ValueError` |
| **API** | external service (Tavily/Gemini/MCP) | `ConnectionError`, `TimeoutError`, `503` |
| **result → serialize** | Python → JSON back out | `TypeError` (e.g. non-serializable object) |

**Quick error decoder (the 8 you'll really see):**

```
TypeError         → wrong type
ValueError        → right type, bad value
KeyError          → dictionary key missing
IndexError        → list position missing
AttributeError    → object doesn't have that attribute
JSONDecodeError   → invalid JSON
FileNotFoundError → file missing
ModuleNotFoundError → Python can't find the module
```

**Rule of the flow:** every arrow is a place things break — so wrap each outside-world call (`DB`/`LLM`/`API`) in `try/except` and return a `{"error": ...}` dict instead of crashing. That's exactly the `@tool` pattern you already write (`except Exception as e: return f"Error ... {e}"`).

## Agent memory: checkpointer + thread_id (remembering conversations)

> *To make an agent **remember previous messages** you need two things: a **checkpointer** (memory storage) + a **`thread_id`** (the conversation's ID). Without them, every `invoke` starts from zero — stateless.*

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model, tools=tools,
    checkpointer=InMemorySaver(),   # << the memory
)

# turn 1 — same thread_id = same conversation
agent.invoke({"messages": [("user", "Hi, I'm Imane")]},
             config={"configurable": {"thread_id": "pfa-1"}})

# turn 2 — the agent REMEMBERS turn 1
agent.invoke({"messages": [("user", "What's my name?")]},
             config={"configurable": {"thread_id": "pfa-1"}})
# → "Imane"
```

### The rules

| You do | Result |
|---|---|
| `checkpointer` + same `thread_id` | agent **remembers everything** in that thread |
| new `thread_id` | **fresh conversation** (no memory) |
| no checkpointer | **forgets** on every call (default) |
| checkpointer but no `thread_id` → error | you must always pass it |

### 3 things to know

1. **`thread_id` is just a string** — use the user's ID (`"pfa-1"` = user 1's chat), so every user gets their own ongoing conversation. That's the setup for the law chatbot.
2. **`InMemorySaver` forgets on restart** (RAM only). For production → SQLite/Postgres checkpointer so memory survives restarts.
3. Once you have a checkpointer, the **middleware** superpowers unlock: `SummarizationMiddleware` to keep tokens flat, time-travel, etc.

### Does it remember only the last message? NO — EVERYTHING

The checkpointer snapshots the **entire state** (all messages + custom fields) at the end of every step, saved under the `thread_id`. Next call with the same id → it restores the whole history, then appends.

**Prove it — read the whole stored thread:**

```python
state = agent.get_state(config={"configurable": {"thread_id": "pfa-1"}})
for m in state.values["messages"]:
    print(m.type, "→", m.content[:40])
# prints ALL messages from all turns, not just the last one
```

**The trade-off:** full memory = the model sees all history again each turn → more tokens (cost rises). That's why the economy move is: **checkpointer (remembers everything) + SummarizationMiddleware (compresses the middle)** → remembers everything, but in a short summary instead of a growing transcript.

**One-liner:** *no checkpointer = amnesia; checkpointer + thread_id = memory; same id = same chat — and it remembers EVERY message in that thread, forever.*

## The 4 token-saving families (+ Trim explained)

> *All token-saving moves belong to 4 families. Biggest win = the recurring bill = **tools + system + history × every turn** — so summarize-then-cache beats any one-off trick.*

### 1. Shrink what you SEND (context management)

| Move | Saves tokens by | Your tool |
|---|---|---|
| **Trim** | deleting old messages | `trim_messages(..., max_tokens=...)` or transformer middleware |
| **Summarize/compaction** | replacing a huge transcript with ~300 token summary | `SummarizationMiddleware` / Claude `/compact` |
| **Tool-result clearing** | dropping bulky re-fetchable outputs, keeping the call record | Claude context-editing; your `@tool` returns stay small |
| **Tight prompts & tool descriptions** | tool schemas + system prompt are re-sent **every** turn — shorter = constant savings | write mini descriptions, not essays |
| **Small tool outputs** | tools that return `dict`s / short strings instead of full DB dumps | your `str` returns from the wedding agents |

These attack the **recurring** cost: system + tools + history re-sent on every message.

### 2. Architecture (stop sending what's irrelevant)

- **Sub-agents = context isolation.** Only the *specialist* receives the file/DB/tool it needs; the coordinator keeps a tiny context (the tiered wedding design). A sub-agent scanning big documents doesn't stuff them into the coordinator.
- **Router first.** A cheap model classifies ("travel? law? venue?") → only the right pipeline runs → the LLM never reads unrelated tools/docs.
- **RAG instead of stuffing.** Send 3 relevant chunks, not the 400-page code penal.

### 3. Model & caching (pay less per token)

- **Small model for 90% of turns** (`gemini-3.1-flash-lite`), escalate to a big model only for hard steps (a cascade).
- **Prompt caching.** The FAQ prefix (system + tools + summary) is cached → cached reads billed at a tiny fraction. Claude does this automatically; Gemini needs manual *context caching*; `cache=` if your provider supports it.
- **Cap `max_tokens`** (answers are part of the window; long reasoning/tool loops cost). Limit thinking budgets if the model exposes them.

### 4. Lifecycle

- **Batch API** for non-urgent work (~2× cheaper).
- **Stream** doesn't save tokens, but makes latency feel instant (UX, not cost).
- **Persist outside** (store/DB) so you never re-derive or re-ask what's already known.

### Cheat sheet for your stack

```
cheapest →  small model throughout
          + trim/summarize history   (keeps per-turn cost FLAT)
          + sub-agents for heavy reads
          + RAG for big legal docs
          + prompt caching if available
```

### Zoom: "Trim = delete" — what it actually does

**Trim just cuts old messages out of the list BEFORE it's sent.** That's all. The model never sees them again. The tokens they'd cost every turn: gone.

```
messages (state)                  →  sent to model this turn
───────────────────────────────────────────────────────────
[sys, h1, ai, h2, ai, h3, ai,     →  [sys, h3, ai, h4, ai]   ← old ones DELETED
 h4, ai, h5, ai]      (10 items)      (5 items, max_tokens budget)
```

Without trim → 10 items in every call · With trim → 5 items in every call = **half the recurring cost**.

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(messages,
                        token_counter="approximate",
                        max_tokens=2000,      # budget: keep ~2k tokens
                        strategy="last",      # keep the newest
                        include_system=True)  # never delete the system prompt
```

Or as agent middleware (runs before every model call):

```python
def trim(state, runtime):
    if len(state["messages"]) > 15:
        return {"messages": state["messages"][-15:]}  # keep last 15

agent = create_agent(model=model, tools=tools,
                     middleware=[AgentMiddleware(transformers=[trim])])
```

**The cost of trimming = it FORGETS.** The 5 deleted messages' information is gone, permanently (no summary). So the choice is:

| | Keep | Lose | When |
|---|---|---|---|
| **Trim** | recent messages | the old ones entirely (**forgets**) | old facts don't matter / want cheapest |
| **Summarize** | recent messages **+ ~300-token summary** of everything | only the exact wording (**remembers compressed**) | you still need to know WHAT was decided |

**One-liner:** *Trim = delete (cheap, forgets); Summarize = compress (slightly pricier, remembers). Keep the system prompt safe; keep the recent window; budget the middle.*

## Working with JSON in Python, `requests` & Flask (the full reference)

> **The 3-second truth:** JSON is **text**. Nothing else. A JSON *string* has no methods, no objects — you must translate it into Python object first (`loads`), and translate your Python object back into a JSON string (`dumps`) before you send it. Every table below is just "translate here vs there".

### Table 1 — the stdlib `json` module (the 4 verbs + 2 flags)

| Call | Direction | When you use it |
|---|---|---|
| `json.loads(text)` | **JSON string → Python** | parsing an API response's `.text` / a Flask body / a file's content |
| `json.dumps(obj)` | **Python → JSON string** | building a payload to send, saving a dict, logging |
| `json.load(f)` | JSON **file → Python** | `f = open("t.json"); data = json.load(f)` |
| `json.dump(obj, f)` | Python → JSON **file** | writing data to a `.json` file |
| `json.dumps(obj, indent=2)` | pretty-print flag | readable dumps |
| `json.dumps(obj, default=str)` | rescue flag | makes `datetime`/objects serializable instead of crashing |

**Memory key:** `load**s**` = **l**oad **s**tring → Python object. `dump**s**` = **d**ump **s**tring ← Python object.

### Table 2 — the Python ↔ JSON translation table

| JSON | appears in Python as (after `loads`) |
|---|---|
| `{}` object | `dict` |
| `[]` array | `list` |
| `"text"` string | `str` |
| `12` / `12.5` number | `int` / `float` |
| `true` / `false` | `bool` (capitalized: `True`/`False`) |
| `null` | `None` |

**The 3 gotchas that break beginners:**
- **tuple** → `dumps` accepts it (becomes array), but `loads` gives back a **list**, not a tuple.
- **set** → cannot serialize at all (`TypeError`). Convert to list first.
- **datetime** → cannot serialize (`TypeError`). Fix: `json.dumps(obj, default=str)`.

### Table 3 — sending JSON with the `requests` library

| Task | Code |
|---|---|
| POST with a JSON body | `resp = requests.post(url, json={"query": "..."})` |
| GET with query params | `resp = requests.get(url, params={"q": "..."})` |
| read the JSON answer | `data = resp.json()`  (now it's a dict — Table 2 rules apply) |
| check it worked | `resp.status_code` ; `resp.raise_for_status()` (raises on 4xx/5xx) |
| response isn't JSON | `resp.json()` → `JSONDecodeError` |

**Note:** `json=` argument does two things automatically — `json.dumps()` + sets `Content-Type: application/json`. If you instead pass `data="..."` (a raw string), *nothing* is automatic: no Content-Type header, and you must make the JSON string yourself.

### Table 4 — receiving JSON in Flask (server side)

| Need | Code |
|---|---|
| get the body as a dict | `body = request.get_json()`  (`None` if body isn't JSON) |
| send a JSON response | `from flask import jsonify` then `return jsonify({"answer": ...})` |
| send JSON + status code | `return jsonify({"error": "..."}), 422` |
| check a body actually IS JSON | `if request.is_json:` |
| read a header | `token = request.headers.get("Authorization")` |
| route values | `/mail/<msg_id>` → `def view(msg_id):` |
| Flask's automatic rejection | non-JSON body → `415`, malformed JSON → `400` (it raises `BadRequest`) |

### Table 5 — the safe pattern (exactly your law-chatbot style)

| Rule | Code |
|---|---|
| validate BEFORE indexing | `if not body or "query" not in body: return jsonify({"error": "missing query"}), 400` |
| wrap outside-world calls | `try: ... except Exception as e: return jsonify({"error": str(e)}), 500` |
| never trust client types | validate/coerce every field instead of using it blindly |
| keep the response small | `jsonify({"answer": text})`, don't echo back the whole request |

**FastAPI one-liner (for later):** same idea, but your route declares a Pydantic `class Body(BaseModel): query: str` as the parameter — validation happens at the door‑frame, and returning a plain dict is converted to JSON automatically.

**The 4-arrow loop (ties to section 19):** client `requests.post(json=...)` → Flask `request.get_json()` → logic → `return jsonify(...)` → client `resp.json()`. Each arrow must translate; JSONDecodeError/TypeError live exactly on those arrows.