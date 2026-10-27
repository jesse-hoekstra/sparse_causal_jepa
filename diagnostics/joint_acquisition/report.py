"""Conservative decision report backed by recorded diagnostic measurements."""

from typing import Any


def _use_parameters(assessment: dict[str, Any]) -> set[int]:
    passing: set[int] = set()
    for row in assessment["per_parameter"]:
        response = row["normalized_response_error"]["interval_95"]
        benefit = row["predictive_benefit"]["interval_95"]
        if response is not None and benefit is not None and response[1] < 1 and benefit[0] > 0:
            passing.add(int(row["parameter"]))
    return passing


def _comparison_observations(comparison: dict[str, Any]) -> list[str]:
    if comparison["status"] != "measured":
        return [
            "The compatible source-code comparison is unavailable; predictor adequacy "
            "remains unresolved."
        ]
    differences = comparison["paired_source_minus_current_ordinary_mse_by_horizon"]
    lines = [
        f"Compatible source-minus-current ordinary MSE by horizon: {differences}. "
        "Negative favors the source; these point differences alone have no reported "
        "confidence interval.",
        "",
        "| Parameter | Source-current response error | Source-current predictive benefit |",
        "| --- | --- | --- |",
    ]
    source_advantage: list[int] = []
    for row in comparison["paired_per_parameter"]:
        error = row["source_minus_current_normalized_response_error"]
        benefit = row["source_minus_current_predictive_benefit"]
        lines.append(
            f"| theta_{row['parameter']} | {error['estimate']}; "
            f"95% CI {error['interval_95']} | {benefit['estimate']}; "
            f"95% CI {benefit['interval_95']} |"
        )
        if (
            error["interval_95"] is not None
            and benefit["interval_95"] is not None
            and error["interval_95"][1] < 0
            and benefit["interval_95"][0] > 0
        ):
            source_advantage.append(int(row["parameter"]))
    lines.append("")
    if source_advantage:
        lines.append(
            f"Paired source-code advantage for parameters {source_advantage}: both "
            "response-error reduction and predictive-benefit gain exclude zero. This "
            "supports a code-dependent limitation; investigate code content and the "
            "predictor around current codes without attributing the gap uniquely to the encoder."
        )
    else:
        lines.append(
            "No parameter has paired intervals establishing both source-code improvements. "
            "The compatible comparison is inconclusive for that joint criterion; ordinary "
            "forecast point differences alone do not resolve it."
        )
    both_unestablished = sorted(
        set(range(1, 7))
        - (_use_parameters(comparison["source"]) | _use_parameters(comparison["current"]))
    )
    if both_unestablished:
        lines.append(
            "Neither code meets both predictive-use bounds, so predictor adequacy remains "
            f"open for parameters {both_unestablished}. Failing these bounds is not proof "
            "of predictor incapacity, and no universal forecast-error cutoff is assumed."
        )
    lines.append("The source encoder remains an imperfect comparator, not an exact oracle.")
    return lines


def _case_observations(result: dict[str, Any]) -> list[str]:
    baseline_pass = _use_parameters(result["baseline"])
    lines: list[str] = []
    accessible: list[str] = []
    for case, local in result["shared_corrections"]["cases"].items():
        if not local.get("defined", False):
            lines.append(f"- {case}: local correction undefined ({local.get('reason')}).")
            continue
        full = local["finite_steps"][0]
        newly_passing = sorted(_use_parameters(full["assessment"]) - baseline_pass)
        incentive = result["objective_incentives"]["cases"][case]
        total = incentive["components"]["total"]["classification"]
        adam = incentive["saved_optimizer_alignment"]["classification"]
        lines.append(
            f"- {case}: full local step newly meets both use bounds for {newly_passing}; "
            f"ordinary total-objective direction is {total}, saved-Adam alignment is {adam}. "
            f"Finite check valid: {local['numerically_valid']}."
        )
        if newly_passing and local["numerically_valid"]:
            accessible.append(case)
            if total == "opposing" or adam == "opposing":
                lines.append(
                    f"  For {case}, the measured local conflict motivates a matched "
                    "objective/optimizer follow-up before changing the production loss."
                )
        trajectory = result["trajectories"][case]
        records = trajectory.get("records", [])
        if records:
            first, final = records[0], records[-1]
            gained = sorted(
                _use_parameters(final["assessment"]) - _use_parameters(first["assessment"])
            )
            changes = [
                round(
                    end["normalized_response_error"]["estimate"]
                    - start["normalized_response_error"]["estimate"],
                    6,
                )
                if end["normalized_response_error"]["estimate"] is not None
                and start["normalized_response_error"]["estimate"] is not None
                else None
                for start, end in zip(
                    first["assessment"]["per_parameter"],
                    final["assessment"]["per_parameter"],
                    strict=True,
                )
            ]
            lines.append(
                f"  At recorded update {final['step']}, response-error point-estimate changes "
                f"are {changes}; newly passing use bounds: {gained}. "
                "These differences are descriptive, not paired improvement intervals."
            )
            if gained and not newly_passing:
                lines.append("  Delayed useful response motivates a bounded timescale follow-up.")
    if accessible == ["joint"]:
        lines.append(
            "Only joint local repair newly passes a use bound: investigate a coordinated "
            "intervention under the reported per-block budgets."
        )
    elif accessible:
        lines.append(
            f"Shared local accessibility candidates: {', '.join(accessible)}. Check their "
            "ordinary forecast errors and trajectories before selecting a follow-up."
        )
    else:
        lines.append(
            "No local case newly meets both use bounds. The next intervention remains "
            "unresolved; inspect trajectory changes and numerical conditioning before "
            "any larger bounded search."
        )
    return lines


def decision_report(result: dict[str, Any]) -> str:
    """Separate observations, numerical validity, interpretation and next decisions."""
    baseline = result["baseline"]
    lines = [
        "# Joint acquisition diagnostic",
        "",
        f"Identity: `{result['identity']}`. Status: {result['scientific_status']}.",
        "",
        "## Observations",
        "",
        "Validation measurements are conditional on one checkpoint and training seed.",
        (
            "The source and current codes are individual-context full codes; "
            "retention and predictive use are separate."
        ),
        "",
        "| Parameter | Response error | Predictive benefit |",
        "| --- | ---: | ---: |",
    ]
    for row in baseline["per_parameter"]:
        lines.append(
            f"| theta_{row['parameter']} | {row['normalized_response_error']['estimate']} "
            f"| {row['predictive_benefit']['estimate']} |"
        )
    lines.extend(
        [
            "",
            f"Ordinary prediction MSE by horizon: {baseline['ordinary_mse_by_horizon']}.",
            f"Validation-selected retention R²: {result['retention']['selected_validation_r2']}.",
            (
                "These probe scores reuse validation for selection and assessment; they "
                "are not final test results."
            ),
            "",
            "## Numerical validity",
            "",
            f"Shared finite checks passed: {result['shared_corrections']['numerically_valid']}.",
            f"Compatible-code control: {result['compatible_control']['status']}.",
            f"Code correction: {result['code_repair']['status']}.",
            (
                "Undefined reference scales or failed finite checks cannot support an "
                "accessibility claim."
            ),
            "",
            "## Local interpretation and next decisions",
            "",
        ]
    )
    lines.extend(_comparison_observations(result["compatible_control"]))
    code = result["code_repair"]
    if code["status"] == "measured":
        improved = [str(parameter) for parameter in sorted(_use_parameters(code["after"]))]
        lines.append(
            "Bounded per-donor repair meets both use bounds for parameters: "
            f"{', '.join(improved) or 'none'}. Transfer across query episodes supports "
            "local code accessibility, not shared acquisition."
        )
    for case, trajectory in result["trajectories"].items():
        lines.append(
            f"- {case}: {trajectory.get('status', 'recorded')}; inspect its recorded "
            "milestones for delayed held-out changes and cumulative movement."
        )
    lines.extend(_case_observations(result))
    lines.extend(
        [
            "",
            (
                "Encoder-only gains motivate checking why ordinary encoder updates miss "
                "useful changes. Predictor-only gains indicate improved use of unchanged "
                "codes, never improved retention. Joint gains where isolated cases fail "
                "motivate a coupled intervention; joint moves both blocks and has a "
                "larger total movement budget."
            ),
            (
                "Useful changes opposed by component or saved-Adam directions motivate a "
                "targeted objective/optimizer control. Aligned directions with delayed "
                "gains motivate a bounded timescale investigation. No improvement after "
                "the full budget is failure within that search and budget, not global "
                "incapacity."
            ),
            "",
            "## Unresolved questions",
            "",
            (
                "No production repair is selected automatically. Inspect raw per-group "
                "uncertainty, actual finite changes and objective incentives before "
                "proposing a matched controlled follow-up. Software checks and supervised "
                "checkpoint repair cannot establish fresh acquisition."
            ),
            (
                "The follow-up order is: evidence-backed intervention; fresh joint "
                "encoder/predictor training with informative states fixed; transfer to "
                "learned states; final dense/baseline/full comparison. Keep one training "
                "seed initially. Do not infer a freezing effect from historical 30k joint "
                "versus 100k frozen runs."
            ),
            "",
            (
                "All numerical settings and source/code identities are in protocol.json. "
                "results.json contains full estimates and intervals; measurements.pt "
                "contains raw group measurements and correction directions. No final-test "
                "measurements are used by this diagnostic."
            ),
        ]
    )
    return "\n".join(lines) + "\n"
