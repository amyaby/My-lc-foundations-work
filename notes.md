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