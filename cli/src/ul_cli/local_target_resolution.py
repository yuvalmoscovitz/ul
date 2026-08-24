from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ul.local_target as local_target_module
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from ul.local_target import (
    CommandTargetConfig,
    LocalTargetConfig,
    LocalTargetConnection,
    PythonCallableTargetConfig,
    create_local_target_dry_run_plan,
    load_local_target_config,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TargetArtifactIdentity(_StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TargetEnvironmentIdentity(_StrictModel):
    name: str
    value_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LocalTargetConfirmation(_StrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["python_callable", "command"]
    reference: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    executable: TargetArtifactIdentity
    artifacts: tuple[TargetArtifactIdentity, ...]
    environment: tuple[TargetEnvironmentIdentity, ...] = ()
    callable: str | None = None

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResolvedLocalTarget:
    reference: str
    config: LocalTargetConfig
    config_sha256: str
    confirmation: LocalTargetConfirmation
    create_connection: Callable[[int, float | None], LocalTargetConnection]
    revalidate_identity: Callable[[], None]

    @property
    def kind(self) -> Literal["python_callable", "command"]:
        return self.config.kind

    @property
    def confirmation_sha256(self) -> str:
        return self.confirmation.sha256

    @property
    def maximum_executions(self) -> int:
        return self.config.limits.max_executions

    @property
    def maximum_active_target_seconds(self) -> float:
        return self.config.limits.total_execution_timeout_seconds


def resolve_local_target(
    reference: str,
    *,
    explicit_artifacts: tuple[Path, ...] = (),
    working_directory: Path | None = None,
    interpreter: Path | None = None,
    environment_allowlist: tuple[str, ...] = (),
) -> ResolvedLocalTarget:
    path = Path(reference)
    if path.is_file():
        if working_directory is not None or interpreter is not None or environment_allowlist:
            raise ValueError("local target options cannot override a target configuration file")
        config = load_local_target_config(path)
    else:
        if ":" not in reference:
            raise ValueError(
                "local target must be module:callable or a local target configuration JSON file"
            )
        config = PythonCallableTargetConfig(
            target_id="probe-" + hashlib.sha256(reference.encode()).hexdigest()[:16],
            working_directory=(working_directory or Path.cwd()).resolve(),
            interpreter=(interpreter or Path(sys.executable)).resolve(),
            target=reference,
            environment_allowlist=environment_allowlist,
        )
    try:
        plan = create_local_target_dry_run_plan(config)
        confirmation = _local_target_confirmation(
            reference, config, plan.config_sha256, explicit_artifacts
        )
    except (OSError, RuntimeError, ValidationError, ValueError) as error:
        raise ValueError(str(error)) from None

    def revalidate_identity() -> None:
        try:
            current = _local_target_confirmation(
                reference, config, plan.config_sha256, explicit_artifacts
            )
        except (OSError, ValueError):
            current = None
        if current != confirmation:
            raise ValueError("confirmed executable or target artifact changed before launch")

    def create_connection(
        maximum_calls: int,
        maximum_seconds: float | None,
    ) -> LocalTargetConnection:
        if maximum_calls > config.limits.max_executions:
            raise ValueError(
                f"target allows at most {config.limits.max_executions} executions; "
                "reduce the campaign or raise limits in its reviewed configuration"
            )
        requested_seconds = (
            maximum_seconds
            if maximum_seconds is not None
            else config.limits.total_execution_timeout_seconds
        )
        if requested_seconds > config.limits.total_execution_timeout_seconds:
            raise ValueError(
                "campaign target time exceeds the reviewed local target execution limit"
            )
        return LocalTargetConnection.from_config(
            config.model_copy(
                update={
                    "limits": config.limits.model_copy(
                        update={
                            "max_executions": maximum_calls,
                            "total_execution_timeout_seconds": requested_seconds,
                        }
                    )
                }
            ),
            customer_code_execution_confirmed=True,
        )

    return ResolvedLocalTarget(
        reference=reference,
        config=config,
        config_sha256=plan.config_sha256,
        confirmation=confirmation,
        create_connection=create_connection,
        revalidate_identity=revalidate_identity,
    )


def local_target_evidence_receipt(target: ResolvedLocalTarget) -> dict[str, JsonValue]:
    confirmation = target.confirmation
    outcome_projection = target.config.outcome
    return {
        "kind": confirmation.kind,
        "config_sha256": confirmation.config_sha256,
        "confirmation_sha256": target.confirmation_sha256,
        "supports_state_observation": False,
        "executable_sha256": confirmation.executable.sha256,
        "artifact_sha256": [artifact.sha256 for artifact in confirmation.artifacts],
        "environment": [item.model_dump(mode="json") for item in confirmation.environment],
        "callable": confirmation.callable,
        "outcome_projection": (
            outcome_projection.model_dump(mode="json") if outcome_projection is not None else None
        ),
        "outcome_projection_sha256": (
            outcome_projection.digest if outcome_projection is not None else None
        ),
    }


def _artifact_identity(path: Path) -> TargetArtifactIdentity:
    resolved = path.resolve(strict=True)
    path_status = os.lstat(resolved)
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_size > 256 * 1024 * 1024
            or not os.path.samestat(path_status, descriptor_status)
        ):
            raise ValueError("target artifact must be a stable bounded regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return TargetArtifactIdentity(path=str(resolved), sha256=digest.hexdigest())


def _resolve_executable(value: Path | str, working_directory: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute() and candidate.parent != Path("."):
        candidate = working_directory / candidate
    if candidate.parent == Path("."):
        found = shutil.which(str(candidate))
        if found is None:
            raise ValueError("target executable was not found")
        candidate = Path(found)
    return candidate.resolve(strict=True)


def _python_module_artifacts(
    config: PythonCallableTargetConfig,
    explicit_artifacts: tuple[Path, ...],
) -> tuple[TargetArtifactIdentity, ...]:
    module_name = config.target.partition(":")[0]
    current = config.working_directory.resolve(strict=True)
    paths: list[Path] = []
    parts = module_name.split(".")
    for index, part in enumerate(parts):
        module_file = current / f"{part}.py"
        package = current / part
        final = index == len(parts) - 1
        if final and module_file.is_file():
            paths.append(module_file)
            break
        init_file = package / "__init__.py"
        if not init_file.is_file():
            raise ValueError("Python target module must resolve to source inside working_directory")
        paths.append(init_file)
        current = package
    paths.append(Path(local_target_module.__file__).with_name("_local_worker.py"))
    paths.extend(
        path if path.is_absolute() else config.working_directory / path
        for path in explicit_artifacts
    )
    identities: dict[str, TargetArtifactIdentity] = {}
    for path in paths:
        identity = _artifact_identity(path)
        identities[identity.path] = identity
    return tuple(identities.values())


def _command_artifacts(
    config: CommandTargetConfig,
    explicit_artifacts: tuple[Path, ...],
) -> tuple[TargetArtifactIdentity, ...]:
    executable = _resolve_executable(config.argv[0], config.working_directory)
    artifacts = [_artifact_identity(executable)]
    for argument in config.argv[1:]:
        candidate = Path(argument)
        if not candidate.is_absolute():
            candidate = config.working_directory / candidate
        if candidate.is_file():
            identity = _artifact_identity(candidate)
            if identity not in artifacts:
                artifacts.append(identity)
    for path in explicit_artifacts:
        candidate = path if path.is_absolute() else config.working_directory / path
        identity = _artifact_identity(candidate)
        if identity not in artifacts:
            artifacts.append(identity)
    generic_prefixes = ("bash", "node", "python", "ruby", "sh")
    if executable.name.startswith(generic_prefixes) and len(artifacts) == 1:
        raise ValueError(
            "generic command workers require --target-artifact for their script or artifact"
        )
    return tuple(artifacts)


def _local_target_confirmation(
    reference: str,
    config: LocalTargetConfig,
    digest: str,
    explicit_artifacts: tuple[Path, ...],
) -> LocalTargetConfirmation:
    if isinstance(config, PythonCallableTargetConfig):
        executable = _artifact_identity(
            _resolve_executable(config.interpreter, config.working_directory)
        )
        artifacts = _python_module_artifacts(config, explicit_artifacts)
        callable_name = config.target
    else:
        artifacts = _command_artifacts(config, explicit_artifacts)
        executable = artifacts[0]
        callable_name = None
    return LocalTargetConfirmation(
        kind=config.kind,
        reference=reference,
        config_sha256=digest,
        executable=executable,
        artifacts=artifacts,
        environment=tuple(
            TargetEnvironmentIdentity(
                name=name,
                value_sha256=hashlib.sha256(os.environ[name].encode()).hexdigest(),
            )
            for name in sorted(config.environment_allowlist)
        ),
        callable=callable_name,
    )
