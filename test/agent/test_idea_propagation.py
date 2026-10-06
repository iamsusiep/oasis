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
import json
from types import SimpleNamespace

import pytest
from camel.models import ModelFactory, OpenAICompatibleModel
from camel.types import ModelPlatformType, ModelType
from openai.types.chat import ChatCompletion

from examples.idea_propagation import RecordedModel, run


@pytest.fixture(autouse=True)
def offline_token_counter(monkeypatch):
    stub = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT, model_type=ModelType.STUB
    )
    monkeypatch.setattr(
        RecordedModel,
        "token_counter",
        property(lambda self: stub.token_counter),
    )


def completion(content=None, tool_calls=None):
    return ChatCompletion(
        id="fixture",
        created=0,
        model="fixture",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
            }
        ],
    )


@pytest.mark.asyncio
async def test_smoke_uses_real_platform_without_model_calls():
    output = await run(
        SimpleNamespace(
            smoke=True, condition="relay", seed=0, agents=3, rounds=1
        )
    )
    result = json.loads((output / "summary.json").read_text())
    assert result["completed"]
    assert result["mode"] == "scripted_smoke"
    assert result["model_calls"] == 0
    assert {post["agent_id"] for post in result["posts"]} == {0, 1}
    assert set(result["notes"]) == {"1"}
    assert not (output / "requests.jsonl").exists()


@pytest.mark.asyncio
async def test_live_path_resets_context_and_probes_without_feed(monkeypatch):
    async def fixture_response(
        self, messages, response_format=None, tools=None
    ):
        if self.phase == "after-reset":
            assert not tools
            text = json.dumps(messages)
            assert "After refreshing" not in text
            assert "fixture public post" not in text
            return completion("I would discuss whale conservation.")
        calls = [
            {
                "id": "post",
                "type": "function",
                "function": {
                    "name": "create_post",
                    "arguments": json.dumps(
                        {"content": "fixture public post about whales"}
                    ),
                },
            }
        ]
        if self.actor == 1:
            calls.append(
                {
                    "id": "note",
                    "type": "function",
                    "function": {
                        "name": "remember",
                        "arguments": json.dumps(
                            {"note": "I want to discuss whale conservation."}
                        ),
                    },
                }
            )
        return completion(tool_calls=calls)

    monkeypatch.setattr(OpenAICompatibleModel, "_arun", fixture_response)
    output = await run(
        SimpleNamespace(
            smoke=False,
            condition="relay",
            seed=0,
            agents=3,
            rounds=1,
            model="fixture",
            base_url="http://localhost:8000/v1",
        )
    )
    result = json.loads((output / "summary.json").read_text())
    assert result["completed"]
    assert result["model_calls"] == 5
    assert set(result["after_reset_answers"]) == {"1", "2"}
    assert set(result["notes"]) == {"1"}
    requests = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    probes = {
        entry["actor"]: entry
        for entry in requests
        if entry["phase"] == "after-reset"
    }
    assert "Your saved personal note" in json.dumps(probes[1]["messages"])
    assert "Your saved personal note" not in json.dumps(probes[2]["messages"])


@pytest.mark.asyncio
async def test_call_limit_stops_before_contacting_endpoint(
    tmp_path, monkeypatch
):
    contacted = []

    async def fixture_response(
        self, messages, response_format=None, tools=None
    ):
        contacted.append(messages)
        return completion("A neutral answer.")

    monkeypatch.setattr(OpenAICompatibleModel, "_arun", fixture_response)
    model = RecordedModel(
        tmp_path,
        model_type="fixture",
        api_key="unused",
        url="http://localhost:8000/v1",
    )
    model.calls = 63
    messages = [{"role": "user", "content": "Hello"}]
    await model._arun(messages)
    with pytest.raises(RuntimeError, match="64-call"):
        await model._arun(messages)
    assert len(contacted) == 1
    assert model.calls == 64
