"""Report generation: text and LaTeX output."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structural_fuzzing.pipeline import StructuralFuzzReport


def format_report(report: StructuralFuzzReport) -> str:
    """Generate a text summary of the campaign results.

    Parameters
    ----------
    report : StructuralFuzzReport
        Complete campaign results.

    Returns
    -------
    str
        Formatted text report.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("STRUCTURAL FUZZING REPORT")
    lines.append("=" * 70)

    # Overview
    lines.append(f"\nDimensions: {len(report.dim_names)}")
    lines.append(f"Dimension names: {', '.join(report.dim_names)}")
    lines.append(f"Total configurations evaluated: {len(report.subset_results)}")

    # Best configuration
    if report.subset_results:
        best = report.subset_results[0]
        lines.append("\nBest configuration:")
        lines.append(f"  Dimensions: {', '.join(best.dim_names)} (k={best.n_dims})")
        lines.append(f"  MAE: {best.mae:.4f}")
        lines.append("  Errors:")
        for name, err in sorted(best.errors.items()):
            lines.append(f"    {name}: {err:+.4f}")

    # Pareto frontier
    lines.append(f"\nPareto frontier ({len(report.pareto_results)} points):")
    for pr in report.pareto_results:
        names = ", ".join(pr.dim_names)
        lines.append(f"  k={pr.n_dims}: MAE={pr.mae:.4f} [{names}]")

    # Sensitivity ranking
    lines.append("\nSensitivity ranking:")
    for sr in report.sensitivity_results:
        lines.append(
            f"  {sr.importance_rank}. {sr.dim_name}: "
            f"delta_MAE={sr.delta_mae:+.4f} "
            f"(with={sr.mae_with:.4f}, without={sr.mae_without:.4f})"
        )

    # MRI
    if report.mri_result is not None:
        mri = report.mri_result
        lines.append("\nModel Robustness Index:")
        lines.append(f"  MRI = {mri.mri:.4f}")
        lines.append(f"  Mean omega = {mri.mean_omega:.4f}")
        lines.append(f"  P75 omega = {mri.p75_omega:.4f}")
        lines.append(f"  P95 omega = {mri.p95_omega:.4f}")
        lines.append(f"  Worst-case MAE = {mri.worst_case_mae:.4f}")
        lines.append(f"  Perturbations = {mri.n_perturbations}")

    # Adversarial thresholds
    lines.append(f"\nAdversarial thresholds ({len(report.adversarial_results)} found):")
    for ar in report.adversarial_results:
        lines.append(
            f"  {ar.dim_name} ({ar.direction}): "
            f"{ar.base_value:.4f} -> {ar.threshold_value:.4f} "
            f"(ratio={ar.threshold_ratio:.2f}x, flips '{ar.target_flipped}')"
        )

    # Compositional test
    if report.composition_result is not None:
        comp = report.composition_result
        lines.append("\nCompositional test:")
        lines.append(f"  Build order: {' -> '.join(comp.order_names)}")
        for i, (name, mae) in enumerate(zip(comp.order_names, comp.mae_sequence)):
            dims_so_far = " + ".join(comp.order_names[: i + 1])
            lines.append(f"  Step {i + 1}: +{name} => MAE={mae:.4f} [{dims_so_far}]")

    # Baselines
    if report.forward_results:
        lines.append(f"\nForward selection ({len(report.forward_results)} steps):")
        for fr in report.forward_results:
            names = ", ".join(fr.dim_names)
            lines.append(f"  k={fr.n_dims}: MAE={fr.mae:.4f} [{names}]")

    if report.backward_results:
        lines.append(f"\nBackward elimination ({len(report.backward_results)} steps):")
        for br in report.backward_results:
            names = ", ".join(br.dim_names)
            lines.append(f"  k={br.n_dims}: MAE={br.mae:.4f} [{names}]")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def format_latex_tables(report: StructuralFuzzReport) -> str:
    """Generate LaTeX tables from campaign results.

    Parameters
    ----------
    report : StructuralFuzzReport
        Complete campaign results.

    Returns
    -------
    str
        LaTeX table code.
    """
    lines: list[str] = []

    # Pareto frontier table
    lines.append("% Pareto Frontier")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Pareto-optimal configurations}")
    lines.append("\\label{tab:pareto}")
    lines.append("\\begin{tabular}{clr}")
    lines.append("\\toprule")
    lines.append("$k$ & Dimensions & MAE \\\\")
    lines.append("\\midrule")
    for pr in report.pareto_results:
        names = ", ".join(pr.dim_names)
        lines.append(f"{pr.n_dims} & {names} & {pr.mae:.4f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Sensitivity table
    lines.append("% Sensitivity Ranking")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Sensitivity profile (ablation)}")
    lines.append("\\label{tab:sensitivity}")
    lines.append("\\begin{tabular}{clrrr}")
    lines.append("\\toprule")
    lines.append("Rank & Dimension & $\\Delta$MAE & MAE (with) & MAE (without) \\\\")
    lines.append("\\midrule")
    for sr in report.sensitivity_results:
        lines.append(
            f"{sr.importance_rank} & {sr.dim_name} & "
            f"{sr.delta_mae:+.4f} & {sr.mae_with:.4f} & {sr.mae_without:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # MRI table
    if report.mri_result is not None:
        mri = report.mri_result
        lines.append("% Model Robustness Index")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Model Robustness Index}")
        lines.append("\\label{tab:mri}")
        lines.append("\\begin{tabular}{lr}")
        lines.append("\\toprule")
        lines.append("Metric & Value \\\\")
        lines.append("\\midrule")
        lines.append(f"MRI & {mri.mri:.4f} \\\\")
        lines.append(f"Mean $\\omega$ & {mri.mean_omega:.4f} \\\\")
        lines.append(f"P75 $\\omega$ & {mri.p75_omega:.4f} \\\\")
        lines.append(f"P95 $\\omega$ & {mri.p95_omega:.4f} \\\\")
        lines.append(f"Worst-case MAE & {mri.worst_case_mae:.4f} \\\\")
        lines.append(f"Perturbations & {mri.n_perturbations} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    return "\n".join(lines)
