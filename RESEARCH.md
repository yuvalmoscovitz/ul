# Trace ingestion research

This is a docs-only compatibility assessment, not a tested integration claim. “Mapping” means a
small exporter or event-to-OTLP translation that emits the allowlisted fields accepted by
`ul dataset ingest otlp --mapping ...`.

## Primary standards

- [OTLP JSON encoding](https://opentelemetry.io/docs/specs/otlp/)
- [OTLP file exporter format](https://opentelemetry.io/docs/specs/otel/protocol/file-exporter/)
- [OpenTelemetry GenAI spans and sensitive-content guidance](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md)

## Agent stack fit

| Stack | Current UL fit | Thin adapter or mapping | Trace/state primitives | Likely customer insight |
|---|---|---|---|---|
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/tracing/) | Medium | Export its agent, generation, function, handoff, and guardrail spans to OTLP GenAI fields | Traces have parented spans; sessions retain history; sensitive trace data is configurable | Tool/handoff paths that turn a safe request into an unsafe side effect |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/event-streaming) | High | Map event-stream messages/tools plus checkpoint values/updates to OTLP messages and state snapshot/delta | Strictly sequenced events, tools, errors, checkpoints, state values and updates | Failure-inducing node/tool sequence and the state transition that made it viable |
| [CrewAI](https://docs.crewai.com/en/concepts/flows) | Medium | Emit Flow listener events and persisted state around existing model/tool telemetry | Event-driven Flow methods, structured state, UUIDs, SQLite/custom persistence | Which crew step or persisted-state carryover introduced a consequential action |
| [AutoGen](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/state.html) | Medium | Map team messages and JSON-serializable `save_state()` snapshots | Agent/team messages, streamed task results, save/load agent and team state | Multi-agent turn where responsibility or context diverged before failure |
| [Google ADK](https://adk.dev/events/) | High | Map session events and `state_delta` to canonical messages/tools/state fields | Sessions, ordered events, tool/function events, event actions with state deltas | The exact event and state delta preceding an unsafe or inconsistent tool choice |
| [PydanticAI](https://pydantic.dev/docs/ai/integrations/logfire/) | High | Configure its OpenTelemetry instrumentation; map message parts if exporter naming differs | Full/new message history, run/conversation IDs, tool parts, OpenTelemetry spans | Broken tool-call/result loops, retry behavior, and conversation-history sensitivity |
| [Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/observability/) | High | Enable GenAI OTel diagnostics and map legacy prompt/completion events or current structured fields | Model spans, function-loop spans, kernel-function spans, parent IDs, opt-in sensitive content | Kernel function or auto-call loop responsible for a high-risk observable difference |
| [LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/agent/agents_as_tools/) | Medium | Export instrumentation/OpenInference plus workflow context snapshots | Workflow events, agent input/output, tool calls/results, context-held history/state | Retrieval/tool trajectory and memory state that produced an unsupported action |
| [Strands Agents](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html) | High | Enable `strands-agents[otel]`; map session baggage and any custom state fields | OTLP GenAI traces with session IDs, parented tool spans, errors and recovery attempts | Tool chain, retry, and span-level origin of a production failure |
| [Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/events-and-streaming) | Medium | Translate persisted session/span/agent events and custom-tool result pairs to OTLP JSON | Versioned agents, sessions, ordered event stream, custom tool use/results, retries/status | Which agent version, event, or unresolved tool action led to an unsafe outcome |

The strongest immediate fits are stacks already producing OTLP GenAI/OpenInference. Event-native
stacks remain viable, but need a small explicit translation; UL should not add vendor-specific
autodetection until real exported fixtures demonstrate stable field contracts.
