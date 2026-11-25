# assistant-iot

A local-first voice assistant in the spirit of a HomePod. Speech recognition, synthesis,
routing and argument extraction all run on-device. The only network calls go to a fixed
allowlist: Open-Meteo for weather, Google's Gmail/Calendar MCP servers, and YouTube for
music. The models never get a browsing tool.

The Jetson Orin Nano Super (8 GB) is the lower bound. Everything also runs on a Mac or a
larger Linux box, and a per-machine profile picks the models.

## How a request flows

```
mic -> wake word / VAD -> speech-to-text -> decision model picks ONE capability (or "other")
    -> small LLM fills that capability's arguments (JSON-schema constrained, then validated)
    -> missing argument? ask.  writes? confirm.  -> flow handler or pinned MCP tool -> speech
```

- **Decision model** (Jev-style "System One", OpenJev-1.7B): answers typed questions with
  probabilities and never generates text. It picks the capability and reads "yes/no"
  confirmations. It also judges whether open-mic speech was meant for the assistant.
- **Small LLM** (Gemma via Ollama or llama.cpp): only extracts arguments and summarises
  results. It never chooses an action, so instructions inside an email can't trigger one.
- **Guards** that do not trust either model:
  - Slots marked `x-verbatim` must come from the user's own words; this blocks invented
    places, names and "now" as a time.
  - Calendar updates only touch an event whose title matches what you named.
  - Every write asks first.

Capabilities live in [config/capabilities.yaml](config/capabilities.yaml). Adding one is a
registry entry, plus a handler in `src/assistant/flows/` for deterministic flows.

| Capability | Backed by | Examples |
|---|---|---|
| weather | Open-Meteo | "will it rain tomorrow", then "what about Saturday?" |
| timer, alarm | on-device scheduler; chimes and speaks until you say anything | "set a pasta timer for 20 minutes", "wake me up at 7" |
| lists | on-device | "add milk and eggs to the shopping list", "what's on my to-do list" |
| memory | on-device notes, found by the closest name | "remember the wifi password is sunflower22", "what's the wifi password" |
| date_time, unit_conversion, math | computed by code, never by the model | "how many days until Christmas", "70 fahrenheit in celsius", "15 percent of 80" |
| read_email, read_calendar, create/update event | Google's MCP servers (off by default, see below) | "what's on my calendar Friday", then "and Saturday?" |
| play_music, music_control | Brave + YouTube | "play some jazz", "turn it up" |
| listening_mode, chat | | "stop listening", "tell me a joke" |

Timers, alarms, lists and notes are JSON files in `~/.local/share/assistant-iot/`, readable
only by you (`storage` in the profile). Clearing a list and forgetting a note ask first.
Follow-ups ("what about Saturday?") work for capabilities marked `follow_up: true`, for 60
seconds after the request.

## Setup (macOS)

```bash
poetry install
```

```bash
poetry run python scripts/fetch_models.py
```

```bash
scripts/run_openjev.sh
```

`fetch_models.py` downloads the speech models (about 300 MB) into `models/` and pins their
hashes in `config/models.lock.json`. `run_openjev.sh` runs the decision model on
127.0.0.1:8092; the first run downloads 3.5 GB. Ollama must be running with `gemma3:12b`;
change `llm.model` in the profile to use another model.

## Run

```bash
poetry run assistant chat
```

```bash
poetry run assistant listen
```

```bash
poetry run assistant mode wake_word
```

```bash
poetry run assistant say "Testing the speaker"
```

- **`chat`:** type instead of talk. Add `-v` to see routing decisions.
- **`listen`:** microphone and speaker. The wake phrase is "hey nova"; change it in the profile.
- **`mode`:** switch a running `listen` between `muted`, `wake_word` and `open_mic`. Only
  this command, or the control socket behind it, can lift `muted`, because muting closes the mic.
- **`say`:** check the speaker and voice.

`--profile fake` runs the whole loop with no models, which is useful for dialogue work.

### Gmail and Calendar (off by default)

Google's MCP servers are Developer Preview. To turn them on:

1. Join the Google Workspace Developer Preview Program.
2. In a Google Cloud project, enable the Gmail API, Gmail MCP API, Google Calendar API and Calendar MCP API.
3. Create an OAuth client of type **Desktop app** and save its JSON to `~/.config/assistant-iot/google-client-secret.json`.
4. Run `poetry run assistant google-login` (opens the browser once; the refresh token goes to the macOS keychain).
5. Check the connection with `poetry run assistant mcp-tools gmail`.
6. Remove the four Google capabilities from `disabled_capabilities` in the profile.

## Measuring

```bash
poetry run pytest
```

```bash
poetry run assistant eval-router --all-capabilities
```

The test suite includes a no-microphone speech round trip: the TTS speaks, then the real
wake-word, VAD and STT models process it. The router eval scores
[tests/routing_eval/utterances.yaml](tests/routing_eval/utterances.yaml): 205 utterances,
including out-of-scope ones, with a threshold sweep.

Measured 2026-11-25 on an M3 Max:

| | Result |
|---|---|
| OpenJev-1.7B routing, 16 capabilities | 89.3% correct at threshold 0.4 (8 wrong routes, 9 misses), ~200 ms per decision. Most wrong routes send calendar questions to date_time, which is read-only. With the first 9 capabilities it was 92.6%. |
| Laya (0.4B) routing, for comparison | ~72%, with confident create-vs-update calendar mistakes |
| Confirmations | every "yes" ≥ 0.70, every "no"/hesitation ≤ 0.32 (threshold 0.6) |
| Moonshine STT | 3.8 s of audio in 0.29 s (2 CPU threads) |
| Piper TTS | 2 s of speech in 0.21 s |

## Swapping models

Each port has adapters chosen in the profile (`config/profiles/*.yaml`):

| Port | Adapters |
|---|---|
| decision | `systemone_http` (OpenJev/wev, laya-serve, llama.cpp decision models, hosted Jev), `fake` |
| llm | `ollama` (loads the model at startup and keeps it loaded), `openai_compat` (llama-server, vLLM, any OpenAI-style server), `fake` |
| speech | sherpa-onnx: `moonshine` / `parakeet` STT, `piper` / `kokoro` TTS, `sherpa_kws`, `silero` |
| weather | `open_meteo`, `fake` |
| music | `brave_youtube`, `fake` |
| tools | `mcp_http` (Streamable HTTP, optional Google OAuth), `fake` |

## Known gaps

- **Alarms:** one-off only; no repeating alarms ("every weekday") or snooze yet.

- **Jetson:** not yet deployed or measured. Two open questions:
  - OpenJev-1.7B runs in fp16 (3.5 GB) through its Python server. Alongside the LLM
    that likely exceeds 8 GB. A llama.cpp GGUF with decision metadata would solve it, but
    the community GGUF lacks that metadata.
  - CUDA wheels for the Jetson's Python version still need checking.
- **Open-mic mode:** the "was that for me?" check is weak on its own. Overheard remarks
  mostly route to "other", and writes always confirm, but expect some false triggers.
- **Music:** the Brave/YouTube player is written and its result-picking is tested, but
  playback is unverified. YouTube barely loads through the corporate network on the dev Mac.
- **Gmail/Calendar:** written against Google's published tool schemas and tested against a
  local MCP server, but not yet against Google's live servers.
- **sherpa-onnx:** pinned to 1.13.x, because 2.x removed the espeak-ng frontend that Piper
  and Kokoro voices need.
