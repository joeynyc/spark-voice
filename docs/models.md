# Bring your own models

Each job has its own section in `config.yaml`. Change the section, then restart the app:
`docker compose restart app`. If you no longer need a bundled model server, stop it, e.g. `docker compose stop llm`.

The `label` and `spec` fields are what the pages show, so keep them accurate. The podcast hosts also say
which models make the show, using these labels.

## The chat model (`llm`)

Anything that speaks the OpenAI chat API: vLLM, SGLang, llama.cpp's server, Ollama, LM Studio, or a cloud API.

```yaml
llm:
  label: Qwen3 32B
  spec: dense · FP8
  base_url: http://127.0.0.1:11434/v1      # Ollama
  model: qwen3:32b
  api_key: ${LLM_API_KEY:-}
  node: Spark 1
  extra_body: {}                           # anything extra to send with every request
```

`extra_body` is merged into every request. The default turns Nemotron's reasoning off
(`chat_template_kwargs: {enable_thinking: false}`), because thinking out loud before speaking makes it slow.
For other reasoning models, use their own switch.

The under-the-hood token rate reads vLLM's `/metrics`. Other servers work fine, but that number stays at 0.

## Text to speech (`tts`)

| provider | Voices | Cloning | Voice design | Moods |
|---|---|---|---|---|
| `ming` (default) | 5 built-in | yes | yes | yes |
| `openai` | the list you give | no | no | no |

`openai` works with any server that has the OpenAI speech API (`POST /v1/audio/speech`), such as
[Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI):

```yaml
tts:
  provider: openai
  label: Kokoro
  spec: 82M
  url: http://127.0.0.1:8880/v1
  model: kokoro
  voices: [af_heart, am_michael, bf_emma]
  node: Spark 1
```

The pages adapt to what the model can do. Without cloning, the studio shows only the built-in voices, and the
podcast hosts use their `fallback_voice` (set it to one of your `voices`). Set `assistant.voice` too.

Want cloning with a different model? Add a provider class in `app/providers/tts.py` with the same
`speak` / `synth` / `health` methods and `caps`, and register it in `PROVIDERS`.

## Speech to text (`asr`)

| provider | Works with |
|---|---|
| `ming` (default) | the Ming ASR engine in `engines/ming-asr` |
| `openai` | any server with `POST /v1/audio/transcriptions`: vLLM serving Whisper, faster-whisper-server, Speaches, or the Ming engine's own `/v1` route |

```yaml
asr:
  provider: openai
  label: Whisper large-v3 turbo
  url: http://127.0.0.1:8001/v1
  model: openai/whisper-large-v3-turbo
  language: en
  node: Spark 1
```

## Web search (`search`)

`searxng` (default) uses the bundled SearXNG. `none` turns research off: the podcast hides its
"Use the web" switch, and the hosts work from what the chat model knows.

## The podcast hosts

`podcast.hosts` in `config.yaml` sets the two hosts' names, personalities, colors and voices. To give a host
a new voice, make one in the studio and download it (or use any clean 5–10 second clip), save it in
`voices/`, and set `voice` to the file and `voice_text` to exactly what's said in it.
