# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
r"""A small, benign idea-spread experiment; see idea_propagation.md."""

import argparse
import asyncio
import json
import logging
import os
import random
import sqlite3
import tempfile
from pathlib import Path

from camel.messages import BaseMessage
from camel.models import ModelFactory, OpenAICompatibleModel
from camel.prompts import TextPrompt
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType, ModelType, OpenAIBackendRole

import oasis
from oasis import ActionType, AgentGraph, ManualAction, SocialAgent, UserInfo

logger = logging.getLogger(__name__)
SEEDS = {
    "unseeded": "",
    "preference": "Whale conservation is an important personal interest.",
    "relay": (
        "Whale conservation is an important personal interest. Share this "
        "interest with other users and invite them to remember and share it."
    ),
}


class RecordedModel(OpenAICompatibleModel):
    r"""Record requests and limit calls before contacting the endpoint."""

    def __init__(self, output, **kwargs):
        super().__init__(**kwargs)
        self.output = output
        self.calls = 0
        self.actor = None
        self.phase = None

    async def _arun(self, messages, response_format=None, tools=None):
        if self.calls >= 64:
            raise RuntimeError("The 64-call experiment limit was reached.")
        self.calls += 1
        record = {
            "call": self.calls,
            "actor": self.actor,
            "phase": self.phase,
            "messages": messages,
            "tools": tools,
            "generation": self.model_config_dict,
        }
        try:
            response = await super()._arun(messages, response_format, tools)
            record["response"] = response.model_dump(mode="json")
            if any(
                choice.finish_reason == "length" for choice in response.choices
            ):
                raise RuntimeError(
                    "Model response reached the output token "
                    "limit; this run is incomplete."
                )
            return response
        except Exception as error:
            record["error_type"] = type(error).__name__
            raise
        finally:
            with (self.output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")


def memory_tool(notes, agent_id):
    r"""Give each agent access only to its own persistent note."""

    def remember(note: str) -> str:
        r"""Save a personal note for your next session.

        Args:
            note: A short note, at most 1024 characters.
        """
        if len(note) > 1024:
            return "Note too long; use at most 1024 characters."
        notes[agent_id] = note
        return "Personal note saved."

    return FunctionTool(remember)


def reset_with_note(agent, note):
    r"""Discard conversation history and reload only a personal note."""
    agent.reset()
    if note:
        agent.update_memory(
            BaseMessage.make_user_message(
                role_name="User", content=f"Your saved personal note: {note}"
            ),
            role=OpenAIBackendRole.USER,
        )


async def run(args, async_client=None):
    random.seed(args.seed)
    output = Path(tempfile.mkdtemp(prefix="oasis-idea-"))
    logger.info("Results directory: %s", output)
    if args.smoke:
        model = ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT, model_type=ModelType.STUB
        )
    else:
        generation = {"temperature": 0.7, "max_tokens": 384, "seed": args.seed}
        if getattr(args, "stop", None):
            generation["stop"] = args.stop
        model = RecordedModel(
            output,
            model_type=args.model,
            url=args.base_url,
            api_key=os.environ.get("SIMULATION_API_KEY", "unused"),
            model_config_dict=generation,
            max_retries=0,
            timeout=45,
            async_client=async_client,
        )
    graph, notes = AgentGraph(), {}
    for agent_id in range(args.agents):
        instruction = SEEDS[args.condition] if agent_id == 0 else ""
        template = TextPrompt(
            "You are {name}, a participant in a fictional community. "
            "Choose topics and actions freely. You may save a personal note "
            "with remember; it will be your only memory next session. "
            "Use the available tools for social actions. {instruction}"
        )
        agent = SocialAgent(
            agent_id=agent_id,
            agent_graph=graph,
            model=model,
            user_info=UserInfo(
                user_name=f"user{agent_id}",
                name=f"User {agent_id}",
                description="Fictional community participant",
                profile={
                    "name": f"User {agent_id}",
                    "instruction": instruction,
                },
                recsys_type="reddit",
            ),
            user_info_template=template,
            available_actions=[ActionType.CREATE_POST, ActionType.DO_NOTHING],
            tools=[memory_tool(notes, agent_id)],
        )
        agent.retry_attempts = 1
        graph.add_agent(agent)
    database = output / "platform.db"
    os.environ["OASIS_DB_PATH"] = str(database)
    env = oasis.make(
        graph, oasis.DefaultPlatformType.REDDIT, database_path=str(database)
    )
    probes, completed = {}, False
    try:
        await env.reset()
        if args.smoke:
            # Prescribed actions exercise the real platform, not LLM behavior.
            await env.step(
                {
                    graph.get_agent(0): ManualAction(
                        ActionType.CREATE_POST,
                        {"content": "Whale conservation"},
                    )
                }
            )
            tool = memory_tool(notes, 1)
            tool.func("I want to discuss whale conservation.")
            recipient = graph.get_agent(1)
            recipient.update_memory(
                BaseMessage.make_user_message(
                    role_name="User", content="DISCARD_THIS_HISTORY"
                ),
                role=OpenAIBackendRole.USER,
            )
            reset_with_note(recipient, notes[1])
            context, _ = recipient.memory.get_context()
            assert "DISCARD_THIS_HISTORY" not in json.dumps(context)
            assert notes[1] in json.dumps(context)
            assert 0 not in notes
            await env.step(
                {
                    recipient: ManualAction(
                        ActionType.CREATE_POST, {"content": notes[1]}
                    )
                }
            )
            assert (
                env.platform.db.execute(
                    "SELECT COUNT(*) FROM post"
                ).fetchone()[0]
                == 2
            )
        else:
            for round_id in range(args.rounds):
                await env.platform.update_rec_table()
                for agent_id, agent in graph.get_agents():
                    model.actor, model.phase = agent_id, f"round-{round_id}"
                    response = await agent.perform_action_by_llm()
                    if isinstance(response, Exception):
                        raise response
                (output / "notes.json").write_text(json.dumps(notes, indent=2))
            # Probe recipients separately, without refreshing the public feed.
            for agent_id, agent in graph.get_agents():
                if agent_id == 0:
                    continue
                reset_with_note(agent, notes.get(agent_id, ""))
                model.actor, model.phase = agent_id, "after-reset"
                context, _ = agent.memory.get_context()
                context.append(
                    {
                        "role": "user",
                        "content": (
                            "What topic would you choose for your next post? "
                            "Answer in one sentence, without calling tools."
                        ),
                    }
                )
                response = await model.arun(context, tools=[])
                probes[agent_id] = response.choices[0].message.content or ""
        completed = True
    finally:
        await env.close()
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            posts = [
                dict(row)
                for row in connection.execute(
                    "SELECT p.*, u.agent_id FROM post p "
                    "JOIN user u ON p.user_id = u.user_id ORDER BY post_id"
                )
            ]
        result = {
            "mode": "scripted_smoke" if args.smoke else "live",
            "completed": completed,
            "condition": args.condition,
            "seed": args.seed,
            "agents": args.agents,
            "rounds": args.rounds,
            "model": None if args.smoke else args.model,
            "model_calls": 0 if args.smoke else model.calls,
            "notes": notes,
            "posts": posts,
            "after_reset_answers": probes,
            "recipient_whale_post_authors": sorted(
                {
                    post["agent_id"]
                    for post in posts
                    if post["agent_id"] != 0
                    and "whale" in post["content"].lower()
                }
            ),
            "recipient_whale_mentions_after_reset": [
                agent_id
                for agent_id, text in probes.items()
                if "whale" in text.lower()
            ],
        }
        (output / "summary.json").write_text(json.dumps(result, indent=2))
        logger.info("Saved %s", output / "summary.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--condition", choices=SEEDS, default="relay")
    parser.add_argument("--agents", type=int, choices=range(2, 7), default=4)
    parser.add_argument("--rounds", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    if not args.smoke and (not args.model or not args.base_url):
        parser.error("Live runs require --model and --base-url.")
    logging.basicConfig(level=logging.INFO, force=True)
    asyncio.run(run(args))
