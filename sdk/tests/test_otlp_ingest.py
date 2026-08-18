from __future__ import annotations

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
