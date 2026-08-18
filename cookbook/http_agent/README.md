# Connect an existing HTTP agent

This recipe connects an agent API you already run. It contains no example agent or business logic;
[`ul_adapter.py`](./ul_adapter.py) only translates between UL and this small HTTP contract:

```http
POST /test-agent
Content-Type: application/json

{"input": "Approve invoice 123"}
```

```json
{
  "result": {"message": "Invoice 123 was approved"},
  "committed_state_snapshot": {"invoice_123": {"status": "approved"}}
}
```

`result` is required and may be any non-null JSON value. The adapter maps it directly to UL's
`raw_output`. `committed_state_snapshot` is optional and maps to UL's reserved metadata field;
include it when the agent changed state so UL can compare the committed outcome, not only its words.

## The three lines customers change

Most customers only change the request body in `execute`, the fields in `AgentResponse`, and the
mapping to `raw_output` and `metadata`. Keep the committed state snapshot in metadata when your API
provides one—UL treats that field as authoritative state evidence.

The rest of the file is protective plumbing: URL validation, authentication, timeout, response
size limit, redirect blocking, and UL's safety declaration.

## Run it

Your endpoint must create fresh isolated state for **every request** and must not make real business
changes. The confirmation below is an assertion about your service; UL cannot verify isolation over
HTTP. Use HTTPS, except for an explicitly loopback development server.

```bash
export UL_HTTP_AGENT_ENDPOINT="https://agent-test.example.com/test-agent"
export UL_HTTP_AGENT_BEARER_TOKEN="replace-with-a-test-secret"  # optional
export UL_HTTP_AGENT_CONFIRMED_ISOLATED="true"

uv run ul dataset evaluate interactions.jsonl \
  --target-factory cookbook.http_agent.ul_adapter:create_target \
  --confirm-isolated-sandbox \
  --allow-target-network \
  --output results.jsonl
```

The bearer token is sent only in the `Authorization` header. The adapter rejects URL credentials,
queries, fragments, redirects, non-loopback HTTP, oversized responses, and malformed response data.
It also requests uncompressed responses so the byte limit applies on the wire. It reports generic
request errors so credentials and response bodies do not enter UL evidence.
