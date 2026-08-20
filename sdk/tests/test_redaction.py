import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError
from ul.environment import evaluation_case_from_inputs
from ul.redaction import (
    LocalPseudonymStore,
    RedactedSemanticPipeline,
    RedactionBoundaryError,
    RedactionEngine,
    RedactionPolicy,
    RedactionRule,
)
from ul_core.dataset import (
    InteractionRecord,
    RenderedUserInput,
    SemanticEquivalenceAssessment,
    SemanticFrame,
    UserInputRecord,
)
from ul_core.evaluation import (
    EnvironmentCapabilities,
    EnvironmentLifecycleEvidence,
    EnvironmentResetEvidence,
    EnvironmentStateEvidence,
    EnvironmentTurnEvidence,
    EvaluationCase,
    ExecutionEvidence,
)

_KEY = SecretStr("a-private-test-key-with-at-least-32-bytes")
_SECRET = "customer@example.com"


def policy() -> RedactionPolicy:
    return RedactionPolicy(
        rules=(
            RedactionRule(
                name="email",
                locations=("input", "output", "context"),
                literal=_SECRET,
            ),
            RedactionRule(
                name="token",
                locations=("output",),
                selector="/credentials/token",
            ),
            RedactionRule(
                name="internal",
                locations=("output",),
                selector="/internal_note",
                action="remove",
            ),
        )
    )


def engine(tmp_path: Path) -> RedactionEngine:
    private_directory = tmp_path / "private"
    private_directory.mkdir(mode=0o700)
    return RedactionEngine(
        policy(), LocalPseudonymStore(private_directory / "pseudonyms.json", _KEY)
    )


def test_pseudonymizes_text_and_json_without_values_in_coverage(tmp_path: Path) -> None:
    redaction = engine(tmp_path)

    result = redaction.transform(
        {
            "message": f"Contact {_SECRET} twice: {_SECRET}",
            "credentials": {"token": "secret-token"},
            "internal_note": "never send this",
        },
        location="output",
    )

    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert _SECRET not in serialized
    assert "secret-token" not in serialized
    assert "never send this" not in serialized
    assert result.coverage.matched_values == 4
    assert result.coverage.matched_paths == (
        "/credentials/token",
        "/internal_note",
        "/message",
    )
    assert result.coverage.matches_by_rule == {"email": 2, "internal": 1, "token": 1}
    assert "internal_note" not in result.value
    message = result.value["message"]
    assert isinstance(message, str)
    placeholders = [word.rstrip(":") for word in message.split() if word.startswith("__UL_")]
    assert placeholders[0] == placeholders[1]


def test_dry_run_reports_coverage_without_writing_state_or_values(tmp_path: Path) -> None:
    redaction = engine(tmp_path)

    result = redaction.transform(f"Contact {_SECRET}", location="input", dry_run=True)

    assert result.coverage.matched_values == 1
    assert result.coverage.matched_paths == ("",)
    assert not redaction.store.path.exists()
    assert _SECRET not in result.coverage.model_dump_json()


def test_reversible_placeholders_are_deterministic_across_resume_and_threads(
    tmp_path: Path,
) -> None:
    private_directory = tmp_path / "private"
    private_directory.mkdir(mode=0o700)
    state_path = private_directory / "pseudonyms.json"

    def pseudonymize() -> str:
        store = LocalPseudonymStore(state_path, _KEY)
        return store.pseudonymize("email", _SECRET)

    with ThreadPoolExecutor(max_workers=8) as executor:
        placeholders = tuple(executor.map(lambda _: pseudonymize(), range(40)))

    assert len(set(placeholders)) == 1
    resumed_store = LocalPseudonymStore(state_path, _KEY)
    assert resumed_store.pseudonymize("email", _SECRET) == placeholders[0]
    assert resumed_store.rehydrate_text(f"Send to {placeholders[0]}") == f"Send to {_SECRET}"
    numeric_placeholder = resumed_store.pseudonymize("account", 1042)
    assert resumed_store.rehydrate_text(f"Account {numeric_placeholder}") == "Account 1042"
    assert os.stat(state_path).st_mode & 0o077 == 0
    assert _SECRET not in state_path.read_text()


def test_store_fails_closed_for_wrong_key_permissions_tampering_and_unknown_tokens(
    tmp_path: Path,
) -> None:
    redaction = engine(tmp_path)
    placeholder = redaction.store.pseudonymize("email", _SECRET)

    with pytest.raises(RedactionBoundaryError, match="failed closed"):
        LocalPseudonymStore(
            redaction.store.path,
            SecretStr("another-private-key-with-at-least-32-bytes"),
        ).rehydrate_text(placeholder)

    os.chmod(redaction.store.path, 0o644)
    with pytest.raises(RedactionBoundaryError, match="failed closed"):
        redaction.store.rehydrate_text(placeholder)
    os.chmod(redaction.store.path, 0o600)

    unknown = "__UL_SECRET_email_00000000000000000000000000000000__"
    with pytest.raises(RedactionBoundaryError, match="failed closed"):
        redaction.store.rehydrate_text(unknown)


def test_policy_rejects_unsupported_or_irreversible_input_selectors() -> None:
    with pytest.raises(ValidationError, match="RFC 6901"):
        RedactionRule(name="secret", selector="$.secret")
    with pytest.raises(ValidationError, match="reversible pseudonymization"):
        RedactionRule(
            name="secret",
            locations=("input",),
            literal="secret",
            action="remove",
        )
    with pytest.raises(ValidationError, match="literal value"):
        RedactionRule(name="secret")

    literal_rule = RedactionRule(name="safe", literal="(a+)+$")
    assert literal_rule.literal == "(a+)+$"


def test_provider_only_text_remove_uses_an_empty_substring(tmp_path: Path) -> None:
    private_directory = tmp_path / "private-remove"
    private_directory.mkdir(mode=0o700)
    redaction = RedactionEngine(
        RedactionPolicy(
            rules=(
                RedactionRule(
                    name="remove",
                    locations=("output",),
                    literal="private ",
                    action="remove",
                ),
            )
        ),
        LocalPseudonymStore(private_directory / "state.json", _KEY),
    )

    result = redaction.transform("private context", location="output")

    assert result.value == "context"


class _RecordingPipeline:
    def __init__(self) -> None:
        self.records: list[InteractionRecord] = []
        self.rendered_inputs: list[str] = []
        self.rendered_instructions: list[str] = []
        self.verified_inputs: list[tuple[str, str]] = []
        self.mutate_placeholder = False
        self.fail_with_secret = False

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        if self.fail_with_secret:
            raise ValueError(f"unsafe: {_SECRET}")
        if not isinstance(record, InteractionRecord):
            raise AssertionError("test pipeline expects an interaction")
        self.records.append(record)
        return SemanticFrame(
            interaction_id=record.id,
            extractor_version="test",
            metadata={},
        )

    async def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> RenderedUserInput:
        self.rendered_inputs.append(raw_input)
        self.rendered_instructions.append(instruction)
        rendered = (
            raw_input.replace("__UL_SECRET_", "__MUTATED_")
            if self.mutate_placeholder
            else raw_input
        )
        return RenderedUserInput(text=f"Please {rendered}", metadata={})

    async def verify(
        self, source_input: str, candidate_input: str
    ) -> SemanticEquivalenceAssessment:
        self.verified_inputs.append((source_input, candidate_input))
        return SemanticEquivalenceAssessment(
            verdict="equivalent",
            explanation="same",
            verifier_version="test",
        )


class _RecordingEnvironment:
    environment_id = "redaction-test-environment"
    config_sha256 = "0" * 64
    capabilities = EnvironmentCapabilities(
        supports_conversations=True,
        supports_state_observation=True,
        state_observation_authority="environment_self_reported",
        cancellation_guarantee="guaranteed",
    )

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        return len(case.turns)

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        self.inputs.extend(turn.content for turn in case.turns)
        response = {"ok": True, "contact": _SECRET}
        state = {"last_contact": _SECRET}
        return ExecutionEvidence(
            case_id=case.id,
            environment_id=self.environment_id,
            environment_config_sha256=self.config_sha256,
            initial_state=EnvironmentStateEvidence(
                value={"initial_contact": _SECRET},
                authority="environment_self_reported",
            ),
            turns=tuple(
                EnvironmentTurnEvidence(
                    turn_id=turn.id,
                    response=response,
                    state_snapshot=state,
                    state_observation_authority="environment_self_reported",
                )
                for turn in case.turns
            ),
            final_response=response,
            final_state=EnvironmentStateEvidence(
                value=state,
                authority="environment_self_reported",
            ),
            lifecycle=EnvironmentLifecycleEvidence(
                initial_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                cleanup_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                terminal_status="succeeded",
                completed_phases=("execute", "cleanup"),
                delivery="certain",
                cleanup="succeeded",
                environment_state_uncertain=False,
            ),
        )


@pytest.mark.asyncio
async def test_pipeline_is_one_boundary_and_environment_rehydrates(tmp_path: Path) -> None:
    redaction = engine(tmp_path)
    provider = _RecordingPipeline()
    pipeline = RedactedSemanticPipeline(provider, redaction)
    source = InteractionRecord(
        id="record-1",
        raw_input=f"Email {_SECRET}",
        raw_observed_output={
            "credentials": {"token": "secret-token"},
            "internal_note": "private context",
        },
    )
    protected_source = pipeline.protect_record(source)
    assert isinstance(protected_source, InteractionRecord)

    frame = await pipeline.deconstruct(protected_source)
    rendered = await pipeline.render(
        protected_source.raw_input, f"rephrase without exposing {_SECRET}"
    )
    assessment = await pipeline.verify(protected_source.raw_input, rendered.text)
    environment = _RecordingEnvironment()
    protected_evidence = await pipeline.wrap_environment(environment).execute(
        evaluation_case_from_inputs(
            case_id="redaction-case",
            raw_inputs=(rendered.text,),
            max_environment_api_calls=1,
            timeout_seconds=30,
        )
    )

    provider_payloads = json.dumps(
        {
            "records": [record.model_dump(mode="json") for record in provider.records],
            "render": provider.rendered_inputs,
            "instructions": provider.rendered_instructions,
            "verify": provider.verified_inputs,
        }
    )
    assert _SECRET not in provider_payloads
    assert "secret-token" not in provider_payloads
    assert "private context" not in provider_payloads
    assert environment.inputs == [f"Please Email {_SECRET}"]
    assert _SECRET not in protected_evidence.model_dump_json()
    assert "__UL_SECRET_email_" in protected_evidence.model_dump_json()
    for metadata in (frame.metadata, rendered.metadata, assessment.metadata):
        assert metadata == {"redaction_policy_sha256": policy().digest}
        assert _SECRET not in json.dumps(metadata)
    assert _SECRET not in protected_source.model_dump_json()


@pytest.mark.asyncio
async def test_pipeline_rejects_placeholder_mutation_and_sanitizes_errors(tmp_path: Path) -> None:
    provider = _RecordingPipeline()
    pipeline = RedactedSemanticPipeline(provider, engine(tmp_path))

    provider.mutate_placeholder = True
    with pytest.raises(RedactionBoundaryError) as mutation_error:
        await pipeline.render(f"Email {_SECRET}", "rephrase")
    assert _SECRET not in str(mutation_error.value)

    provider.fail_with_secret = True
    with pytest.raises(RedactionBoundaryError) as provider_error:
        await pipeline.deconstruct(
            InteractionRecord(
                id="record-1",
                raw_input=f"Email {_SECRET}",
                raw_observed_output={"credentials": {"token": "secret-token"}},
            )
        )
    assert _SECRET not in str(provider_error.value)
