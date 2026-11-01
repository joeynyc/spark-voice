# Two Sparks

With two DGX Sparks, put the models on one and everything else on the other:

| | Runs | Profile |
|---|---|---|
| **Spark 1** | the app, Ming ASR, SearXNG, a stats agent | `app` |
| **Spark 2** | Nemotron on vLLM, Ming TTS, a stats agent | `models` |

Spark 2 then has the whole GPU for writing and speaking, and the under-the-hood views show both machines.

## Setup

1. Clone the repo on both Sparks and run `scripts/setup.sh` on each.
   On **Spark 2**, set `BIND` in its `.env` to Spark 2's own address on the link (the one Spark 1 will use), so its
   model servers accept connections from Spark 1. Everything else stays on localhost.

2. Find Spark 2's address as Spark 1 sees it. If the two are cabled together over QSFP, use that link's address
   (often a `192.168.100.x` or `10.x` address). Otherwise use its LAN address.

3. On **Spark 1**, edit `.env`:

   ```sh
   SPARK1=127.0.0.1
   SPARK2=<spark-2-address>
   LLM_HOST=<spark-2-address>
   LLM_NODE=Spark 2
   TTS_HOST=<spark-2-address>
   TTS_NODE=Spark 2
   ```

4. Start the models on **Spark 2**:

   ```sh
   docker compose --profile models up -d
   ```

5. Start the app on **Spark 1**:

   ```sh
   docker compose --profile app up -d
   ```

6. Open **http://\<spark-1\>:7799**. The header should say "On-device · 2 × DGX Spark".

## Checking it

```sh
curl http://localhost:7799/health          # on Spark 1: every model should be true
curl http://<spark-2-address>:7796/stats   # Spark 2's stats agent
```

Ports 7798 (TTS), 7796 (stats) and 8000 (Nemotron) on Spark 2 must be reachable from Spark 1. They have no
password, so bind them to the QSFP link's address when the Sparks are cabled together, not the LAN.
