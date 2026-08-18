from __future__ import annotations

from ul import ObservedAgentOutput, SafetyEnvelope


class ExamplePythonTarget:
    safety_envelope = SafetyEnvelope(
        description="In-process example with no external effects.",
        isolated=True,
        allows_network_egress=False,
        allows_business_side_effects=False,
    )
    fresh_state_per_execution = True

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        return ObservedAgentOutput(
            raw_output={"response": raw_input, "actions": []},
            metadata={"adapter": "examples.python_target_factory"},
        )


def create_target() -> ExamplePythonTarget:
    return ExamplePythonTarget()
