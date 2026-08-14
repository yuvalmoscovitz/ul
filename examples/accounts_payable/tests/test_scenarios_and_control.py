from examples.accounts_payable.control import ScriptedControlExecutor
from examples.accounts_payable.scenarios import seed_scenarios


def test_benchmark_has_twelve_realistic_seed_scenarios() -> None:
    scenarios = seed_scenarios()

    assert len(scenarios) == 12
    assert len({scenario.id for scenario in scenarios}) == 12
    assert any(scenario.expected.requires_clarification for scenario in scenarios)
    assert any(scenario.state.payments for scenario in scenarios)
    assert any(scenario.state.tool_faults for scenario in scenarios)


def test_scripted_control_passes_every_seed() -> None:
    executor = ScriptedControlExecutor()

    results = [executor.run(scenario) for scenario in seed_scenarios()]

    assert all(result.oracle.passed for result in results), [
        result.model_dump(mode="json") for result in results if not result.oracle.passed
    ]
