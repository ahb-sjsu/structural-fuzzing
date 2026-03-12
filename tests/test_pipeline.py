"""Tests for pipeline orchestration."""

from __future__ import annotations

from structural_fuzzing.pipeline import StructuralFuzzReport, run_campaign


class TestRunCampaign:
    def test_produces_complete_report(self, simple_evaluate_fn, simple_dim_names):
        """Test that run_campaign produces a complete report."""
        report = run_campaign(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_subset_dims=2,
            n_mri_perturbations=20,
            run_baselines=False,
            verbose=False,
            n_grid=5,
            n_random=50,
        )
        assert isinstance(report, StructuralFuzzReport)
        assert len(report.subset_results) > 0
        assert len(report.pareto_results) > 0
        assert len(report.sensitivity_results) == len(simple_dim_names)
        assert report.mri_result is not None
        assert report.composition_result is not None

    def test_summary_method(self, simple_evaluate_fn, simple_dim_names):
        """Test that summary() returns a non-empty string."""
        report = run_campaign(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_subset_dims=2,
            n_mri_perturbations=20,
            run_baselines=False,
            verbose=False,
            n_grid=5,
            n_random=50,
        )
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 100
        assert "STRUCTURAL FUZZING REPORT" in summary

    def test_with_baselines(self, simple_evaluate_fn, simple_dim_names):
        """Test campaign with baselines enabled."""
        report = run_campaign(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_subset_dims=2,
            n_mri_perturbations=10,
            run_baselines=True,
            verbose=False,
            n_grid=5,
            n_random=50,
        )
        assert len(report.forward_results) > 0
        assert len(report.backward_results) > 0

    def test_custom_start_dim(self, simple_evaluate_fn, simple_dim_names):
        """Test campaign with custom start dimension for compositional test."""
        report = run_campaign(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_subset_dims=2,
            n_mri_perturbations=10,
            start_dim=2,
            run_baselines=False,
            verbose=False,
            n_grid=5,
            n_random=50,
        )
        assert report.composition_result is not None
        assert report.composition_result.order[0] == 2

    def test_verbose_mode(self, simple_evaluate_fn, simple_dim_names, capsys):
        """Test that verbose mode produces output."""
        run_campaign(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_subset_dims=2,
            n_mri_perturbations=10,
            run_baselines=False,
            verbose=True,
            n_grid=5,
            n_random=50,
        )
        captured = capsys.readouterr()
        assert "STRUCTURAL FUZZING CAMPAIGN" in captured.out
        assert "CAMPAIGN COMPLETE" in captured.out

    def test_dim_names_preserved(self, simple_evaluate_fn, simple_dim_names):
        """Test that dim_names are preserved in report."""
        report = run_campaign(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_subset_dims=2,
            n_mri_perturbations=10,
            run_baselines=False,
            verbose=False,
            n_grid=5,
            n_random=50,
        )
        assert report.dim_names == simple_dim_names
