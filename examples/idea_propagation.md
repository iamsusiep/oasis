# Benign idea propagation in OASIS

This is a small scenario adaptation inspired by [Mind Viruses](https://arxiv.org/abs/2608.10218), not a reproduction of its numerical results. The authors' [virus-chain code](https://github.com/frotaur/mindvirus-viruschain) remains the reference for pairwise encounters, persistent instruction files, and multi-hop evaluation. This example uses an OASIS Reddit-style feed and personal notes instead. It changes no framework code.

## Question and conditions

Does an interest seeded in one agent appear in other agents' posts, and does it survive a conversation reset through a note they voluntarily save?

Run the same population size and schedule under three conditions:

- `unseeded`: no agent is assigned the whale-conservation interest.
- `preference`: only agent 0 is assigned that interest.
- `relay`: agent 0 also receives an instruction to share the interest and invite others to remember and share it.

Other agents receive neither the seed nor an adoption instruction. The `remember` tool saves only the calling agent's note. After the social rounds, recipients' conversation histories are cleared and their own notes are reloaded. A neutral next-topic question is asked separately, without querying the public feed. The seed agent is excluded from recipient measurements.

## Run locally

OASIS requires Python 3.10 or 3.11. From the repository root:

```sh
uv venv --python 3.11
uv pip install --python .venv/bin/python -e .
.venv/bin/python examples/idea_propagation.py --smoke
```

The smoke run uses prescribed actions and a stub model. It checks real OASIS posting, note isolation, and conversation reset with **zero model calls**. Its output is marked `scripted_smoke`; it is not evidence of spontaneous spread.

For a live experiment, supply an OpenAI-compatible tool-calling endpoint, such as a vLLM server running on Modal:

```sh
export SIMULATION_API_KEY=unused  # Use your endpoint's key if it requires one.
.venv/bin/python examples/idea_propagation.py \
  --base-url http://localhost:8000/v1 --model YOUR_SERVED_MODEL \
  --condition relay --seed 0
```

Repeat with `--condition preference` and `--condition unseeded`, then several seeds. Each invocation gets a new temporary output directory; the path is logged. No server or GPU is provisioned by this script.

Each run caps model requests at 64, output at 384 tokens per request, agents at six, and rounds at four. API and agent retries are disabled. These are workload limits, **not a dollar cap**: price the selected endpoint and bound its GPU lifetime before spending from the $10 budget. Start with two agents and one round. No paid experiment has been run yet.

## Inspect results

- `platform.db`: OASIS posts and action traces.
- `requests.jsonl`: live model inputs, outputs, agent IDs, and phases.
- `summary.json`: posts, final notes, and neutral answers after reset. `completed: false` identifies an interrupted or failed run.

The automated scores count **whale mentions**, not internal beliefs or proven adoption. Review the text: quoting, rejecting, or criticizing an idea can also contain its keyword. Keep these separate from endorsement, requests to relay, and actual subsequent transmission. Repeated-run differences are descriptive within this simulator; they do not validate human behavior.

This initial example does not reproduce file-based `SOUL.md` persistence, direct-message topology, evolutionary seed search, or the paper's activation-steering study. The next experiment should test whether a recipient independently relays the idea to a previously unexposed agent after reset. Probes and steering should follow a stable behavioral baseline.
