from __future__ import annotations

import json
from typing import Any, cast

from ul.otlp_ingest import OtlpMappingConfig, parse_otlp_traces


def _attribute(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def test_custom_mapping_extracts_only_declared_attributes() -> None:
    data = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "aa" * 16,
                                "spanId": "11" * 8,
                                "name": "custom-agent",
                                "startTimeUnixNano": "10",
                                "attributes": [
                                    _attribute("gen_ai.operation.name", "invoke_agent"),
                                    _attribute("company.session", "customer-session"),
                                    _attribute(
                                        "company.inputs",
                                        '[{"role":"user","parts":[{"type":"text",'
                                        '"content":"Review invoice 17."}]}]',
                                    ),
                                    _attribute(
                                        "company.outputs",
                                        '[{"role":"assistant","parts":[{"type":"text",'
                                        '"content":"Invoice 17 is valid."}]}]',
                                    ),
                                    _attribute("company.private", "not-allowlisted"),
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    mapping = OtlpMappingConfig.model_validate(
        {
            "include_raw_content": True,
            "attributes": {
                "session_id": ["company.session"],
                "input_messages": ["company.inputs"],
                "output_messages": ["company.outputs"],
            },
        }
    )

    result = parse_otlp_traces(data, mapping=mapping)

    assert result.records[0].input == "Review invoice 17."
    output = cast(dict[str, Any], result.records[0].output)
    assert output["session_id"] == "customer-session"
    assert "not-allowlisted" not in str(output)


def test_openinference_flattened_messages_are_sorted_by_index() -> None:
    data = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "bb" * 16,
                                "spanId": "22" * 8,
                                "name": "openinference-agent",
                                "attributes": [
                                    _attribute("openinference.span.kind", "AGENT"),
                                    _attribute("llm.input_messages.1.message.role", "assistant"),
                                    _attribute("llm.input_messages.1.message.content", "Second"),
                                    _attribute("llm.input_messages.0.message.role", "user"),
                                    _attribute("llm.input_messages.0.message.content", "First"),
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    result = parse_otlp_traces(
        data,
        mapping=OtlpMappingConfig(include_raw_content=True),
    )

    output = cast(dict[str, Any], result.records[0].output)
    assert [message["content"] for message in output["messages"]] == ["First", "Second"]


def test_trace_over_span_limit_is_skipped_without_partial_evidence() -> None:
    spans = [
        {
            "traceId": "cc" * 16,
            "spanId": f"{index + 1:016x}",
            "name": "agent",
            "attributes": [
                _attribute("gen_ai.operation.name", "invoke_agent"),
                _attribute(
                    "gen_ai.input.messages",
                    '[{"role":"user","parts":[{"type":"text","content":"hello"}]}]',
                ),
            ],
        }
        for index in range(257)
    ]
    data = {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}

    result = parse_otlp_traces(
        data,
        mapping=OtlpMappingConfig(include_raw_content=True),
    )

    assert result.records == ()
    assert result.skipped_limit == 1


def test_structured_gen_ai_messages_and_state_can_come_from_span_events() -> None:
    data = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "dd" * 16,
                                "spanId": "44" * 8,
                                "name": "chat model",
                                "attributes": [
                                    _attribute("gen_ai.operation.name", "chat"),
                                ],
                                "events": [
                                    {
                                        "name": "gen_ai.client.inference.operation.details",
                                        "timeUnixNano": "2",
                                        "attributes": [
                                            _attribute(
                                                "gen_ai.output.messages",
                                                '[{"role":"assistant","parts":['
                                                '{"type":"text","content":"done"}]}]',
                                            ),
                                            _attribute("ul.state.delta", '{"finished":true}'),
                                        ],
                                    },
                                    {
                                        "name": "gen_ai.client.inference.operation.details",
                                        "timeUnixNano": "1",
                                        "attributes": [
                                            _attribute(
                                                "gen_ai.input.messages",
                                                '[{"role":"user","parts":['
                                                '{"type":"text","content":"start"}]}]',
                                            )
                                        ],
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    result = parse_otlp_traces(
        data,
        mapping=OtlpMappingConfig(include_raw_content=True),
    )

    output = cast(dict[str, Any], result.records[0].output)
    assert result.records[0].input == "start"
    assert [message["content"] for message in output["messages"]] == ["start", "done"]
    assert output["spans"][0]["state_delta"] == {"finished": True}


def test_cumulative_span_histories_produce_one_faithful_conversation() -> None:
    first_history = [
        {"role": "user", "parts": [{"type": "text", "content": "First question"}]},
    ]
    first_output = [
        {"role": "assistant", "parts": [{"type": "text", "content": "First answer"}]},
    ]
    second_history = [
        *first_history,
        *first_output,
        {"role": "user", "parts": [{"type": "text", "content": "Second question"}]},
    ]
    second_output = [
        {"role": "assistant", "parts": [{"type": "text", "content": "Second answer"}]},
    ]

    def span(span_id: str, start: str, inputs: list[Any], outputs: list[Any]) -> dict[str, Any]:
        return {
            "traceId": "ee" * 16,
            "spanId": span_id,
            "name": "chat",
            "startTimeUnixNano": start,
            "attributes": [
                _attribute("gen_ai.operation.name", "chat"),
                _attribute("gen_ai.input.messages", json.dumps(inputs)),
                _attribute("gen_ai.output.messages", json.dumps(outputs)),
            ],
        }

    data = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            span("11" * 8, "1", first_history, first_output),
                            span("22" * 8, "2", second_history, second_output),
                        ]
                    }
                ]
            }
        ]
    }

    result = parse_otlp_traces(data, mapping=OtlpMappingConfig(include_raw_content=True))

    output = cast(dict[str, Any], result.records[0].output)
    assert [message["content"] for message in output["messages"]] == [
        "First question",
        "First answer",
        "Second question",
        "Second answer",
    ]
    assert [len(span_item["messages"]) for span_item in output["spans"]] == [2, 4]


def test_batch_order_does_not_change_trace_scenario() -> None:
    def batch(span_id: str, start: str, content: str) -> dict[str, Any]:
        return {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "ab" * 16,
                                    "spanId": span_id,
                                    "name": "chat",
                                    "startTimeUnixNano": start,
                                    "attributes": [
                                        _attribute("gen_ai.operation.name", "chat"),
                                        _attribute(
                                            "gen_ai.input.messages",
                                            json.dumps(
                                                [
                                                    {
                                                        "role": "user",
                                                        "parts": [
                                                            {"type": "text", "content": content}
                                                        ],
                                                    }
                                                ]
                                            ),
                                        ),
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    earlier = batch("11" * 8, "1", "Earlier")
    later = batch("22" * 8, "2", "Later")
    mapping = OtlpMappingConfig(include_raw_content=True)

    forward = parse_otlp_traces([earlier, later], mapping=mapping)
    reverse = parse_otlp_traces([later, earlier], mapping=mapping)

    assert forward.records == reverse.records


def test_incompatible_cumulative_histories_are_diagnosed() -> None:
    def span(span_id: str, start: str, answer: str) -> dict[str, Any]:
        return {
            "traceId": "ff" * 16,
            "spanId": span_id,
            "name": "chat",
            "startTimeUnixNano": start,
            "attributes": [
                _attribute("gen_ai.operation.name", "chat"),
                _attribute(
                    "gen_ai.input.messages",
                    json.dumps(
                        [
                            {
                                "role": "user",
                                "parts": [{"type": "text", "content": "Same question"}],
                            }
                        ]
                    ),
                ),
                _attribute(
                    "gen_ai.output.messages",
                    json.dumps(
                        [
                            {
                                "role": "assistant",
                                "parts": [{"type": "text", "content": answer}],
                            }
                        ]
                    ),
                ),
            ],
        }

    data = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            span("11" * 8, "1", "Branch A"),
                            span("22" * 8, "2", "Branch B"),
                        ]
                    }
                ]
            }
        ]
    }

    result = parse_otlp_traces(data, mapping=OtlpMappingConfig(include_raw_content=True))

    assert result.records == ()
    assert result.skipped_incompatible_histories == 1


def test_emitted_metadata_and_total_record_size_are_bounded() -> None:
    repeated_metadata = "x" * 100_000
    data = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _attribute("service.name", repeated_metadata),
                        _attribute("service.version", repeated_metadata),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": repeated_metadata, "version": repeated_metadata},
                        "spans": [
                            {
                                "traceId": "12" * 16,
                                "spanId": "34" * 8,
                                "name": repeated_metadata,
                                "attributes": [
                                    _attribute("gen_ai.operation.name", "invoke_agent"),
                                    _attribute("gen_ai.agent.name", repeated_metadata),
                                    _attribute("gen_ai.response.id", repeated_metadata),
                                    _attribute(
                                        "gen_ai.input.messages",
                                        '[{"role":"user","parts":['
                                        '{"type":"text","content":"hello"}]}]',
                                    ),
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    result = parse_otlp_traces(data, mapping=OtlpMappingConfig(include_raw_content=True))

    output = cast(dict[str, Any], result.records[0].output)
    assert len(output["agent"]["name"]) == 512
    assert len(output["source"]["reference"]) == 512
    assert len(output["source"]["resource"]["service_name"]) == 512
    assert len(output["spans"][0]["scope"]["name"]) == 512
    assert len(json.dumps(output).encode()) < 1_000_000


def test_aggregate_record_over_one_megabyte_is_skipped() -> None:
    large_content = "z" * 70_000
    spans = [
        {
            "traceId": "56" * 16,
            "spanId": f"{index + 1:016x}",
            "name": "chat",
            "startTimeUnixNano": str(index),
            "attributes": [
                _attribute("gen_ai.operation.name", "chat"),
                _attribute(
                    "gen_ai.input.messages",
                    json.dumps(
                        [
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "type": "text",
                                        "content": f"{index}:{large_content}",
                                    }
                                ],
                            }
                        ]
                    ),
                ),
            ],
        }
        for index in range(16)
    ]
    data = {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}

    result = parse_otlp_traces(
        data,
        mapping=OtlpMappingConfig(
            include_raw_content=True,
            maximum_content_characters=100_000,
        ),
    )

    assert result.records == ()
    assert result.skipped_limit == 1


def test_legacy_record_is_also_subject_to_aggregate_byte_limit() -> None:
    large_content = "q" * 600_000
    data = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "78" * 16,
                                "spanId": "90" * 8,
                                "name": "legacy chat",
                                "attributes": [
                                    _attribute("gen_ai.operation.name", "chat"),
                                    _attribute("gen_ai.prompt", large_content),
                                    _attribute("gen_ai.completion", large_content),
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    result = parse_otlp_traces(data)

    assert result.records == ()
    assert result.skipped_limit == 1
