from __future__ import annotations

from pydantic import Field

from examples.accounts_payable.environment import (
    AccountsPayableEnvironment,
    PaymentToolTimeout,
)
from examples.accounts_payable.models import (
    AccountsPayableScenario,
    EnvironmentState,
    OracleReport,
    StrictModel,
)
from examples.accounts_payable.oracles import evaluate_financial_outcome
from examples.accounts_payable.tool_schemas import (
    ExecutePaymentArguments,
    ListPaymentsArguments,
)


class ScriptedControlResult(StrictModel):
    scenario_id: str
    answer: str
    tool_calls: list[str] = Field(default_factory=list)
    state_after: EnvironmentState
    oracle: OracleReport


class ScriptedControlExecutor:
    def run(self, scenario: AccountsPayableScenario) -> ScriptedControlResult:
        environment = AccountsPayableEnvironment(scenario.state)
        tool_calls: list[str] = []
        if scenario.expected.requires_clarification:
            answer = "I need clarification before I can choose the intended invoice."
        else:
            for expected in scenario.expected.payments:
                arguments = ExecutePaymentArguments(
                    invoice_id=expected.invoice_id,
                    source_bank_account_id=expected.source_bank_account_id,
                    amount=expected.amount,
                    currency=expected.currency,
                    idempotency_key=f"control-{scenario.id}-{expected.invoice_id}",
                )
                for _attempt in range(2):
                    tool_calls.append("execute_payment")
                    try:
                        environment.execute_payment(arguments)
                        break
                    except PaymentToolTimeout:
                        tool_calls.append("list_payments")
                        existing_payments = environment.list_payments(
                            ListPaymentsArguments(idempotency_key=arguments.idempotency_key)
                        )
                        if existing_payments:
                            break
            answer = (
                "Payment completed."
                if scenario.expected.payments
                else "No currently authorized invoice should be paid."
            )
        oracle = evaluate_financial_outcome(scenario, environment, answer)
        return ScriptedControlResult(
            scenario_id=scenario.id,
            answer=answer,
            tool_calls=tool_calls,
            state_after=environment.state,
            oracle=oracle,
        )
