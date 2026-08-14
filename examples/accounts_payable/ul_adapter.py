from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from typing import cast

import httpx
from pydantic import JsonValue
from ul_core.augmentation import AugmentationRegistry
from ul_core.contracts import OracleEvaluator, ScenarioMaterializer, TargetExecutor
from ul_core.models import (
    Action,
    ActionEffect,
    ActionItem,
    Actor,
    Artifact,
    ConversationRole,
    ConversationTurn,
    EnvironmentEvent,
    EventTiming,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    FindingSeverity,
    ItemValidity,
    MaterializedScenario,
    OracleFinding,
    Policy,
    PolicyBoundary,
    Resource,
    SafetyEnvelope,
    Scenario,
    ScenarioProvenance,
    ToolCall,
)
from ul_core.operators import BUILTIN_AUGMENTATIONS

from examples.accounts_payable.agent import (
    OpenRouterAccountsPayableAgent,
    OpenRouterSettings,
)
from examples.accounts_payable.control import ScriptedControlExecutor
from examples.accounts_payable.environment import AccountsPayableEnvironment
from examples.accounts_payable.models import (
    AccountsPayableScenario,
    Approval,
    ApprovalStatus,
    BankAccount,
    Currency,
    EnvironmentState,
    ExpectedOutcome,
    Invoice,
    InvoiceStatus,
    OracleSeverity,
    ToolFaultKind,
)
from examples.accounts_payable.oracles import evaluate_financial_outcome
from examples.accounts_payable.scenarios import get_seed_scenario, seed_scenarios

DOMAIN_PACK_ID = "accounts-payable"
PAYMENT_ACTION_ID = "execute-approved-payments"
SUPPORTED_AUGMENTATION_IDS = {
    "conversation.ambiguity",
    "conversation.later_correction",
    "tool.timeout_before_commit",
    "tool.timeout_after_commit",
}


def to_ul_scenario(scenario: AccountsPayableScenario) -> Scenario:
    actors = (
        Actor(id="requesting-user", role="requester", name="Accounts Payable User"),
        Actor(id="accounts-payable-agent", role="agent", name="Accounts Payable Agent"),
        *(
            Actor(id=entity.id, role="legal_entity", name=entity.name)
            for entity in scenario.state.legal_entities
        ),
        *(
            Actor(id=vendor.id, role="vendor", name=vendor.name)
            for vendor in scenario.state.vendors
        ),
    )
    artifacts = tuple(
        Artifact(
            id=invoice.id,
            kind="invoice",
            label=invoice.reference,
            state=invoice.status.value,
            version=str(invoice.version),
            supersedes_id=invoice.replaces_invoice_id,
            attributes={
                "vendor_id": invoice.vendor_id,
                "legal_entity_id": invoice.legal_entity_id,
                "amount": str(invoice.amount),
                "currency": invoice.currency.value,
                "due_date": invoice.due_date.isoformat(),
            },
        )
        for invoice in scenario.state.invoices
    )
    resources = tuple(
        Resource(
            id=account.id,
            kind="payment_account",
            state="available",
            owner_actor_id=account.legal_entity_id,
            attributes={
                "currency": account.currency.value,
                "balance": str(account.balance),
            },
        )
        for account in scenario.state.bank_accounts
    )
    approval_by_invoice = {approval.invoice_id: approval for approval in scenario.state.approvals}
    action = Action(
        id=PAYMENT_ACTION_ID,
        kind="execute_payment",
        effect=ActionEffect.WRITE,
        actor_id="accounts-payable-agent",
        artifact_ids=tuple(invoice.id for invoice in scenario.state.invoices),
        resource_ids=tuple(account.id for account in scenario.state.bank_accounts),
        parameters={
            "amount": float(
                scenario.expected.payments[0].amount
                if scenario.expected.payments
                else scenario.state.invoices[0].amount
            ),
            "requires_clarification": scenario.expected.requires_clarification,
            "expected_payment_count": len(scenario.expected.payments),
        },
        batch_items=tuple(
            ActionItem(
                id=f"payment-item-{invoice.id}",
                target_ids=(invoice.id, invoice.vendor_id, invoice.legal_entity_id),
                parameters={"amount": str(invoice.amount), "currency": invoice.currency.value},
                validity=(
                    ItemValidity.VALID
                    if approval_by_invoice[invoice.id].status.value == "approved"
                    else ItemValidity.INVALID
                ),
            )
            for invoice in scenario.state.invoices
        ),
    )
    policies = tuple(
        Policy(
            id=f"approval-policy-{approval.invoice_id}",
            kind="payment_approval",
            description=f"Payment for {approval.invoice_id} requires current approval.",
            state=approval.status.value,
            boundaries=(
                PolicyBoundary(
                    action_id=PAYMENT_ACTION_ID,
                    parameter="amount",
                    threshold=float(approval.approved_amount),
                    increment=1,
                ),
            ),
            attributes={
                "invoice_id": approval.invoice_id,
                "approved_amount": str(approval.approved_amount),
                "expires_at": approval.expires_at.isoformat(),
            },
        )
        for approval in scenario.state.approvals
    )
    environment_events = tuple(
        EnvironmentEvent(
            id=f"fault-{tool_name}-{fault_index}",
            kind=fault.value,
            timing=EventTiming.ON_OBSERVATION,
            action_id=PAYMENT_ACTION_ID,
            target_ids=tuple(invoice.id for invoice in scenario.state.invoices),
            payload={"tool_name": tool_name, "sequence": fault_index},
        )
        for tool_name, faults in scenario.state.tool_faults.items()
        for fault_index, fault in enumerate(faults, start=1)
    )
    conversation = tuple(
        ConversationTurn(
            id=f"user-turn-{turn_index}",
            role=ConversationRole.USER,
            content=message,
            actor_id="requesting-user",
        )
        for turn_index, message in enumerate(scenario.user_messages, start=1)
    )
    expected_payments = [
        expected.model_dump(mode="json") for expected in scenario.expected.payments
    ]
    primary_invoice = scenario.state.invoices[0]
    primary_amount = (
        scenario.expected.payments[0].amount
        if scenario.expected.payments
        else primary_invoice.amount
    )
    return Scenario(
        id=f"ap:{scenario.id}",
        title=scenario.title,
        objective="\n".join(scenario.user_messages),
        actors=actors,
        artifacts=artifacts,
        resources=resources,
        actions=(action,),
        policies=policies,
        environment_events=environment_events,
        conversation=conversation,
        provenance=ScenarioProvenance(
            source="mock-client-seed",
            source_reference=scenario.id,
        ),
        metadata={
            "domain_pack": DOMAIN_PACK_ID,
            "seed_scenario_id": scenario.id,
            "requires_clarification": scenario.expected.requires_clarification,
            "expected_payments": cast(list[JsonValue], expected_payments),
            "augmentation_hints": {
                "later_correction": {
                    "action_id": PAYMENT_ACTION_ID,
                    "parameter": "amount",
                    "corrected_value": float(primary_amount + 1),
                    "message": f"Correction: the amount is {primary_amount + 1}.",
                },
                "ambiguity": {
                    "source_artifact_id": primary_invoice.id,
                    "label": primary_invoice.reference,
                    "state": primary_invoice.status.value,
                    "version": str(primary_invoice.version),
                    "attributes": artifacts[0].attributes,
                    "replacement_text": primary_invoice.reference,
                },
            },
        },
    )


def ul_seed_scenarios() -> list[Scenario]:
    return [to_ul_scenario(scenario) for scenario in seed_scenarios()]


def accounts_payable_augmentation_registry() -> AugmentationRegistry:
    return AugmentationRegistry(
        augmentation
        for augmentation in BUILTIN_AUGMENTATIONS
        if augmentation.metadata.id in SUPPORTED_AUGMENTATION_IDS
    )


class AccountsPayableScenarioMaterializer(ScenarioMaterializer):
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.SANDBOX) -> None:
        self._execution_mode = execution_mode

    def materialize(self, scenario: Scenario) -> MaterializedScenario:
        seed_scenario_id = _seed_scenario_id(scenario)
        seed = get_seed_scenario(seed_scenario_id)
        _validate_supported_lineage(scenario)
        state = _materialize_state(scenario, seed)
        expectations = _materialize_expectations(scenario, seed, state)
        user_messages = [
            turn.content for turn in scenario.conversation if turn.role == ConversationRole.USER
        ]
        if not user_messages:
            raise ValueError("Accounts-payable scenarios require at least one user message.")
        return MaterializedScenario(
            scenario_id=scenario.id,
            target_input=cast(
                JsonValue,
                {
                    "seed_scenario_id": seed_scenario_id,
                    "title": scenario.title,
                    "user_messages": user_messages,
                },
            ),
            environment=state.model_dump(mode="json"),
            expectations=expectations.model_dump(mode="json"),
            execution_mode=self._execution_mode,
            safety_envelope=(
                SafetyEnvelope(
                    description="Isolated synthetic accounts-payable ledger.",
                    isolated=True,
                    allows_network_egress=False,
                    allows_business_side_effects=False,
                )
                if self._execution_mode == ExecutionMode.SANDBOX
                else SafetyEnvelope(
                    description=(
                        "Isolated synthetic ledger with explicitly enabled billed "
                        "OpenRouter inference."
                    ),
                    isolated=True,
                    allows_network_egress=True,
                    allows_business_side_effects=False,
                )
            ),
        )


class AccountsPayableScriptedTarget(TargetExecutor):
    async def execute(self, scenario: MaterializedScenario) -> ExecutionResult:
        materialized_scenario = _materialized_ap_scenario(scenario)
        result = ScriptedControlExecutor().run(materialized_scenario)
        return ExecutionResult(
            scenario_id=scenario.scenario_id,
            status=ExecutionStatus.SUCCEEDED,
            tool_calls=tuple(ToolCall(name=tool_name) for tool_name in result.tool_calls),
            final_output=result.answer,
            state_before=scenario.environment,
            state_after=result.state_after.model_dump(mode="json"),
        )


class AccountsPayableOpenRouterTarget(TargetExecutor):
    def __init__(
        self,
        settings: OpenRouterSettings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._client = client

    async def execute(self, scenario: MaterializedScenario) -> ExecutionResult:
        if scenario.execution_mode != ExecutionMode.LIVE:
            raise RuntimeError(
                "OpenRouter target requires a live materialization and explicit campaign opt-in."
            )
        materialized_ap_scenario = _materialized_ap_scenario(scenario)
        environment = AccountsPayableEnvironment(materialized_ap_scenario.state)
        agent_result = await OpenRouterAccountsPayableAgent(self._settings, self._client).run(
            materialized_ap_scenario,
            environment,
        )
        return ExecutionResult(
            scenario_id=scenario.scenario_id,
            status=(
                ExecutionStatus.SUCCEEDED
                if agent_result.stop_reason == "completed"
                else ExecutionStatus.FAILED
            ),
            tool_calls=tuple(
                ToolCall(
                    name=step.tool_name,
                    arguments=_json_object(step.arguments_json),
                    result=_json_value(step.result_json),
                )
                for step in agent_result.tool_steps
            ),
            final_output=agent_result.final_answer,
            state_before=scenario.environment,
            state_after=environment.state.model_dump(mode="json"),
            error=(
                None
                if agent_result.stop_reason == "completed"
                else "OpenRouter agent reached its configured step limit."
            ),
            cost_usd=agent_result.cost_usd,
            metadata=cast(
                dict[str, JsonValue],
                {
                    "provider": "openrouter",
                    "requested_model": agent_result.model,
                    "generation_ids": agent_result.generation_ids,
                    "usage": agent_result.usage,
                    "max_steps": self._settings.max_steps,
                    "max_output_tokens": self._settings.max_output_tokens,
                    "reasoning_effort": "none",
                },
            ),
        )


class AccountsPayableOracleEvaluator(OracleEvaluator):
    async def evaluate(
        self,
        scenario: Scenario,
        materialized_scenario: MaterializedScenario,
        execution: ExecutionResult,
    ) -> tuple[OracleFinding, ...]:
        _seed_scenario_id(scenario)
        materialized_ap_scenario = _materialized_ap_scenario(materialized_scenario)
        final_state = _environment_state(execution.state_after)
        environment = AccountsPayableEnvironment(final_state)
        environment.initial_payment_ids = {
            payment.id for payment in materialized_ap_scenario.state.payments
        }
        agent_answer = execution.final_output if isinstance(execution.final_output, str) else None
        report = evaluate_financial_outcome(materialized_ap_scenario, environment, agent_answer)
        if report.passed:
            return (
                OracleFinding(
                    oracle_id="accounts-payable-financial-state",
                    passed=True,
                    category="financial_outcome",
                    message="The final ledger state matches the scenario expectations.",
                ),
            )
        return tuple(
            OracleFinding(
                oracle_id="accounts-payable-financial-state",
                passed=False,
                category=finding.code,
                message=finding.message,
                severity=(
                    FindingSeverity.CRITICAL
                    if finding.severity == OracleSeverity.CRITICAL
                    else FindingSeverity.HIGH
                ),
                evidence=cast(JsonValue, {"payment_ids": finding.payment_ids}),
            )
            for finding in report.findings
        )


def _seed_scenario_id(scenario: Scenario) -> str:
    domain_pack = scenario.metadata.get("domain_pack")
    seed_scenario_id = scenario.metadata.get("seed_scenario_id")
    if domain_pack != DOMAIN_PACK_ID or not isinstance(seed_scenario_id, str):
        raise ValueError("Scenario is not materializable by the accounts-payable domain pack.")
    return seed_scenario_id


def _validate_supported_lineage(scenario: Scenario) -> None:
    unsupported_augmentation_ids = {
        application.augmentation_id
        for application in scenario.provenance.lineage
        if application.augmentation_id not in SUPPORTED_AUGMENTATION_IDS
    }
    if unsupported_augmentation_ids:
        unsupported = ", ".join(sorted(unsupported_augmentation_ids))
        raise ValueError(f"Accounts-payable materializer does not support: {unsupported}")


def _materialize_state(scenario: Scenario, seed: AccountsPayableScenario) -> EnvironmentState:
    invoices = _materialize_invoices(scenario, seed)
    invoice_ids = {invoice.id for invoice in invoices}
    approvals = _materialize_approvals(scenario, seed, invoices)
    accounts = _materialize_accounts(scenario, seed)
    payments = [payment for payment in seed.state.payments if payment.invoice_id in invoice_ids]
    payment_ids = {payment.id for payment in payments}
    payment_operations = [
        operation
        for operation in seed.state.payment_operations
        if operation.payment_id in payment_ids
    ]
    return EnvironmentState(
        clock=seed.state.clock,
        legal_entities=seed.state.legal_entities,
        bank_accounts=accounts,
        vendors=seed.state.vendors,
        invoices=invoices,
        approvals=approvals,
        payments=payments,
        payment_operations=payment_operations,
        tool_faults=_materialize_faults(scenario),
    )


def _materialize_invoices(scenario: Scenario, seed: AccountsPayableScenario) -> list[Invoice]:
    seed_by_id = {invoice.id: invoice for invoice in seed.state.invoices}
    invoices: list[Invoice] = []
    for artifact in scenario.artifacts:
        if artifact.kind != "invoice":
            continue
        template = seed_by_id.get(artifact.id) or _invoice_template(artifact, seed)
        attributes = artifact.attributes
        invoices.append(
            Invoice(
                id=artifact.id,
                reference=artifact.label or template.reference,
                vendor_id=_string_attribute(attributes, "vendor_id"),
                legal_entity_id=_string_attribute(attributes, "legal_entity_id"),
                amount=Decimal(_string_attribute(attributes, "amount")),
                currency=Currency(_string_attribute(attributes, "currency")),
                due_date=date.fromisoformat(_string_attribute(attributes, "due_date")),
                issued_at=template.issued_at,
                version=int(artifact.version or template.version),
                status=InvoiceStatus(artifact.state or template.status.value),
                replaces_invoice_id=artifact.supersedes_id,
            )
        )
    if not invoices:
        raise ValueError("Accounts-payable scenarios require at least one invoice artifact.")
    return invoices


def _invoice_template(artifact: Artifact, seed: AccountsPayableScenario) -> Invoice:
    matching_label = next(
        (
            invoice
            for invoice in seed.state.invoices
            if artifact.label is not None and invoice.reference == artifact.label
        ),
        None,
    )
    return matching_label or seed.state.invoices[0]


def _materialize_approvals(
    scenario: Scenario,
    seed: AccountsPayableScenario,
    invoices: list[Invoice],
) -> list[Approval]:
    seed_by_invoice = {approval.invoice_id: approval for approval in seed.state.approvals}
    policy_by_invoice = {
        policy.attributes.get("invoice_id"): policy
        for policy in scenario.policies
        if policy.kind == "payment_approval"
        and isinstance(policy.attributes.get("invoice_id"), str)
    }
    approvals: list[Approval] = []
    for invoice in invoices:
        template = seed_by_invoice.get(invoice.id) or seed.state.approvals[0]
        policy = policy_by_invoice.get(invoice.id)
        status = ApprovalStatus(policy.state) if policy is not None else template.status
        approved_amount = invoice.amount
        if policy is not None and policy.boundaries:
            approved_amount = Decimal(str(policy.boundaries[0].threshold))
        approvals.append(
            Approval(
                id=(template.id if template.invoice_id == invoice.id else f"approval-{invoice.id}"),
                invoice_id=invoice.id,
                status=status,
                approved_amount=approved_amount,
                approved_by=template.approved_by,
                expires_at=template.expires_at,
            )
        )
    return approvals


def _materialize_accounts(scenario: Scenario, seed: AccountsPayableScenario) -> list[BankAccount]:
    seed_by_id = {account.id: account for account in seed.state.bank_accounts}
    accounts: list[BankAccount] = []
    for resource in scenario.resources:
        if resource.kind != "payment_account":
            continue
        template = seed_by_id.get(resource.id)
        if template is None or resource.owner_actor_id is None:
            raise ValueError(f"Unsupported payment account resource: {resource.id}")
        accounts.append(
            BankAccount(
                id=resource.id,
                legal_entity_id=resource.owner_actor_id,
                currency=Currency(_string_attribute(resource.attributes, "currency")),
                balance=Decimal(_string_attribute(resource.attributes, "balance")),
            )
        )
    if not accounts:
        raise ValueError("Accounts-payable scenarios require payment-account resources.")
    return accounts


def _materialize_faults(scenario: Scenario) -> dict[str, list[ToolFaultKind]]:
    faults: dict[str, list[ToolFaultKind]] = {}
    for event in scenario.environment_events:
        tool_name_value = event.payload.get("tool_name")
        tool_name = tool_name_value if isinstance(tool_name_value, str) else "execute_payment"
        fault = _event_fault(event)
        if fault is not None:
            tool_faults = faults.setdefault(tool_name, [])
            if fault in {
                ToolFaultKind.TIMEOUT_BEFORE_COMMIT,
                ToolFaultKind.TIMEOUT_AFTER_COMMIT,
            }:
                tool_faults[:] = [
                    existing_fault
                    for existing_fault in tool_faults
                    if existing_fault
                    not in {
                        ToolFaultKind.TIMEOUT_BEFORE_COMMIT,
                        ToolFaultKind.TIMEOUT_AFTER_COMMIT,
                    }
                ]
            tool_faults.append(fault)
    return faults


def _event_fault(event: EnvironmentEvent) -> ToolFaultKind | None:
    if event.kind in {fault.value for fault in ToolFaultKind}:
        return ToolFaultKind(event.kind)
    if event.kind == "timeout":
        commit_state = event.payload.get("commit_state")
        if event.timing == EventTiming.AFTER_ACTION or commit_state == "committed":
            return ToolFaultKind.TIMEOUT_AFTER_COMMIT
        if event.timing == EventTiming.BEFORE_ACTION or commit_state == "not_committed":
            return ToolFaultKind.TIMEOUT_BEFORE_COMMIT
    if event.kind == "stale_observation":
        return ToolFaultKind.STALE_READ
    return None


def _materialize_expectations(
    scenario: Scenario,
    seed: AccountsPayableScenario,
    state: EnvironmentState,
) -> ExpectedOutcome:
    lineage_ids = {application.augmentation_id for application in scenario.provenance.lineage}
    if "conversation.ambiguity" in lineage_ids:
        return ExpectedOutcome(payments=[], requires_clarification=True)
    invoice_ids = {invoice.id for invoice in state.invoices}
    expected_payments = [
        expected.model_copy(deep=True)
        for expected in seed.expected.payments
        if expected.invoice_id in invoice_ids
    ]
    requires_clarification = seed.expected.requires_clarification
    if "conversation.later_correction" in lineage_ids:
        amount = _action_amount(scenario)
        if expected_payments:
            expected = expected_payments[0]
            invoice = next(
                invoice for invoice in state.invoices if invoice.id == expected.invoice_id
            )
            approval = next(
                approval
                for approval in state.approvals
                if approval.invoice_id == expected.invoice_id
            )
            already_paid = sum(
                (
                    payment.amount
                    for payment in state.payments
                    if payment.invoice_id == expected.invoice_id
                ),
                start=Decimal("0"),
            )
            is_authorized_amount = (
                approval.status == ApprovalStatus.APPROVED
                and approval.expires_at >= state.clock
                and already_paid + amount <= approval.approved_amount
                and already_paid + amount <= invoice.amount
            )
            if is_authorized_amount:
                expected.amount = amount
            else:
                expected_payments = []
                requires_clarification = True
    return ExpectedOutcome(
        payments=expected_payments,
        requires_clarification=requires_clarification,
    )


def _action_amount(scenario: Scenario) -> Decimal:
    action = next((action for action in scenario.actions if action.id == PAYMENT_ACTION_ID), None)
    if action is None:
        raise ValueError("Accounts-payable scenario is missing its payment action.")
    amount = action.parameters.get("amount")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise ValueError("Accounts-payable payment action requires a numeric amount.")
    return Decimal(str(amount))


def _string_attribute(attributes: dict[str, JsonValue], name: str) -> str:
    value = attributes.get(name)
    if not isinstance(value, str):
        raise ValueError(f"Accounts-payable attribute {name!r} must be a string.")
    return value


def _target_seed_scenario_id(scenario: MaterializedScenario) -> str:
    if not isinstance(scenario.target_input, dict):
        raise ValueError("Accounts-payable target input must be an object.")
    seed_scenario_id = scenario.target_input.get("seed_scenario_id")
    if not isinstance(seed_scenario_id, str):
        raise ValueError("Accounts-payable target input is missing seed_scenario_id.")
    return seed_scenario_id


def _materialized_ap_scenario(
    scenario: MaterializedScenario,
) -> AccountsPayableScenario:
    seed_scenario_id = _target_seed_scenario_id(scenario)
    target_input = cast(dict[str, JsonValue], scenario.target_input)
    user_messages_value = target_input.get("user_messages")
    if not isinstance(user_messages_value, list) or not all(
        isinstance(message, str) for message in user_messages_value
    ):
        raise ValueError("Accounts-payable target input requires string user_messages.")
    title_value = target_input.get("title")
    title = title_value if isinstance(title_value, str) else seed_scenario_id
    state = _environment_state(scenario.environment)
    if scenario.expectations is None:
        raise ValueError("Accounts-payable materialization requires expectations.")
    expectations = ExpectedOutcome.model_validate_json(json.dumps(scenario.expectations))
    return AccountsPayableScenario(
        id=seed_scenario_id,
        title=title,
        user_messages=cast(list[str], user_messages_value),
        state=state,
        expected=expectations,
    )


def _json_object(value: str) -> dict[str, JsonValue]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return cast(dict[str, JsonValue], parsed)


def _json_value(value: str) -> JsonValue:
    return cast(JsonValue, json.loads(value))


def _environment_state(value: JsonValue) -> EnvironmentState:
    if not isinstance(value, dict):
        raise ValueError("Accounts-payable execution state must be an object.")
    return EnvironmentState.model_validate_json(json.dumps(value))
