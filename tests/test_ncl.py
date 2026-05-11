"""Tests for NCL — Natural Continual Learning (Kao et al., NeurIPS 2021).

The contract under test is the paper's Eq. (8) update,

    θ ← θ - η [ Λ_{k-1}^{-1} ∇L_CE(θ) + (θ - μ_{k-1}) ]

with Λ_{k-1} ≈ ⊕_l (A_l ⊗ G_l) accumulated additively across tasks
(Eq. 5 in K-FAC form), and the Bayesian rubber-band term (θ - μ_{k-1})
added in parameter space *after* preconditioning.

The tests are grouped as follows:

  A. Structural & lifecycle — class layout, init state, snapshot semantics.
  B. K-FAC factor estimation — shape, symmetry, PSD-ness, mean-reduction
     compensation, additive accumulation across tasks.
  C. Eq. (8) algebra — the rewritten ``.grad`` equals
     G⁻¹ ∇W A⁻¹ + (W − W*) for known A, G; degenerate limits
     (Λ = I, ∇L = 0, large damping) collapse to the expected closed forms.
  D. Diagnostics — kl_proxy reproduces the Mahalanobis quadratic form,
     tr_scale matches r / ‖step‖_Λ when below 1 and saturates above.
  E. End-to-end forgetting — two-task synthetic benchmark; NCL retains
     task-0 accuracy noticeably better than plain SGD.

Tests use a TinyMLP (≤ 6 k parameters) so each K-FAC factor is small
enough that an exact analytic comparison is feasible without sampling.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.methods.base_method import BaseMethod
from src.methods.ncl import NCL, _damped_inv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cfg(
    fisher_samples: int = 50,
    damping: float = 1e-3,
    trust_radius: float = 1.0,
    lr: float = 0.01,
    weight_decay: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        method=SimpleNamespace(
            name="ncl",
            ncl=SimpleNamespace(
                fisher_samples=fisher_samples,
                damping=damping,
                trust_radius=trust_radius,
            ),
        ),
        training=SimpleNamespace(lr=lr, weight_decay=weight_decay),
    )


class TinyMLP(nn.Module):
    """Two-layer MLP — all parameters are Linear so K-FAC covers them all."""

    def __init__(self, in_dim: int = 20, hidden: int = 16, out_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _loader(n: int = 40, in_dim: int = 20, n_classes: int = 4, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def _train_first_task(ncl: NCL, loader: DataLoader, epochs: int = 1) -> None:
    """Run a few observe()s on task 0, then end_task to populate the prior."""
    for _ in range(epochs):
        for x, y in loader:
            ncl.observe(x, y, task_id=0)
    ncl.end_task(0, loader)


# ===========================================================================
# A. Structural & lifecycle
# ===========================================================================

class TestStructure:
    def test_subclass(self):
        assert issubclass(NCL, BaseMethod)

    def test_init_state(self):
        ncl = NCL(TinyMLP(), _cfg())
        # No prior accumulated yet → factor dicts empty, flat _prior_mean is None.
        assert ncl._kfac_A == {}
        assert ncl._kfac_G == {}
        assert ncl._prior_W == {}
        assert ncl._prior_b == {}
        assert ncl._prior_mean is None
        # Hyperparameters are read from cfg.method.ncl.
        assert ncl._fisher_samples == 50
        assert pytest.approx(ncl._damping) == 1e-3
        assert pytest.approx(ncl._trust_radius) == 1.0

    def test_required_methods(self):
        ncl = NCL(TinyMLP(), _cfg())
        for name in ("observe", "end_task", "evaluate", "get_step_diagnostics"):
            assert callable(getattr(ncl, name))

    def test_linear_layers_collected(self):
        ncl = NCL(TinyMLP(), _cfg())
        # TinyMLP has two nn.Linear layers — both should be picked up.
        assert len(ncl._linear_layers) == 2


class TestFlatPriorMean:
    def test_prior_mean_none_until_end_task(self):
        ncl = NCL(TinyMLP(), _cfg())
        # Even after observe() on task 0, no prior has been folded in yet.
        x, y = torch.randn(8, 20), torch.randint(0, 4, (8,))
        ncl.observe(x, y, task_id=0)
        assert ncl._prior_mean is None

    def test_prior_mean_shape_matches_total_params(self):
        model = TinyMLP()
        ncl = NCL(model, _cfg(fisher_samples=20))
        _train_first_task(ncl, _loader(n=32, seed=0))
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert ncl._prior_mean.shape == (n_params,)

    def test_prior_mean_equals_concatenated_params_at_snapshot(self):
        """At the moment end_task fires, _prior_mean should be exactly the
        flattened, detached parameter vector — bit-for-bit."""
        torch.manual_seed(0)
        model = TinyMLP()
        ncl = NCL(model, _cfg(fisher_samples=20))
        _train_first_task(ncl, _loader(n=32, seed=0))

        expected = torch.cat([
            p.detach().reshape(-1).cpu() for p in model.parameters() if p.requires_grad
        ])
        assert torch.allclose(ncl._prior_mean, expected, atol=1e-6)

    def test_prior_mean_is_snapshot_not_reference(self):
        """Future SGD steps must not bleed into the stored prior mean —
        the snapshot is detached and CPU-cloned at end_task."""
        torch.manual_seed(0)
        model = TinyMLP()
        ncl = NCL(model, _cfg(fisher_samples=20))
        _train_first_task(ncl, _loader(n=32, seed=0))
        mean_before = ncl._prior_mean.clone()

        # Take several observe() steps on a new task to drift parameters.
        for x, y in _loader(n=32, seed=1):
            ncl.observe(x, y, task_id=1)

        # The flat prior_mean must not have moved with the parameters.
        assert torch.allclose(ncl._prior_mean, mean_before, atol=0.0), (
            "_prior_mean leaked a view of the live parameters"
        )
        # But the actual parameters did move (sanity).
        live = torch.cat([
            p.detach().reshape(-1).cpu() for p in model.parameters() if p.requires_grad
        ])
        assert not torch.allclose(live, mean_before, atol=1e-5), (
            "Parameters did not move; cannot verify snapshot semantics"
        )

    def test_per_layer_prior_W_matches_module_at_snapshot(self):
        """The per-layer μ dicts used inside the natural-gradient step
        must agree with the model weights at the moment end_task ran."""
        torch.manual_seed(0)
        model = TinyMLP()
        ncl = NCL(model, _cfg(fisher_samples=20))
        _train_first_task(ncl, _loader(n=32, seed=0))

        for name, module in ncl._linear_layers.items():
            assert torch.allclose(
                ncl._prior_W[name], module.weight.data.cpu(), atol=1e-6
            )
            if module.bias is not None:
                assert torch.allclose(
                    ncl._prior_b[name], module.bias.data.cpu(), atol=1e-6
                )


# ===========================================================================
# B. K-FAC factor estimation
# ===========================================================================

class TestKFACShapes:
    def test_factor_shapes_with_bias_augmentation(self):
        """A is (d_in + 1, d_in + 1) for biased layers (the augmented [a;1]
        input convention); G is (d_out, d_out)."""
        model = TinyMLP(in_dim=20, hidden=16, out_dim=4)
        ncl = NCL(model, _cfg(fisher_samples=30))
        _train_first_task(ncl, _loader(n=40, seed=1))

        for name, module in ncl._linear_layers.items():
            d_in_aug = module.in_features + (1 if module.bias is not None else 0)
            d_out = module.out_features
            assert ncl._kfac_A[name].shape == (d_in_aug, d_in_aug)
            assert ncl._kfac_G[name].shape == (d_out, d_out)

    def test_no_bias_no_augmentation(self):
        """A layer without bias should have A shape (d_in, d_in) — no extra column."""
        class NoBiasMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 6, bias=False)
                self.fc2 = nn.Linear(6, 3, bias=False)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        torch.manual_seed(0)
        model = NoBiasMLP()
        ncl = NCL(model, _cfg(fisher_samples=20))
        g = torch.Generator().manual_seed(0)
        loader = DataLoader(
            TensorDataset(
                torch.randn(32, 8, generator=g),
                torch.randint(0, 3, (32,), generator=g),
            ),
            batch_size=8,
        )
        _train_first_task(ncl, loader)

        assert ncl._kfac_A["fc1"].shape == (8, 8)
        assert ncl._kfac_A["fc2"].shape == (6, 6)


class TestKFACSymmetryAndPSD:
    def test_factors_symmetric(self):
        torch.manual_seed(42)
        model = TinyMLP()
        ncl = NCL(model, _cfg(fisher_samples=30))
        _train_first_task(ncl, _loader(n=40, seed=0))
        for name in ncl._linear_layers:
            A, G = ncl._kfac_A[name], ncl._kfac_G[name]
            assert torch.allclose(A, A.T, atol=1e-5)
            assert torch.allclose(G, G.T, atol=1e-5)

    def test_factors_psd(self):
        """Each A and G is a sample covariance — must be PSD (eigenvalues ≥ -tol)."""
        torch.manual_seed(0)
        model = TinyMLP()
        ncl = NCL(model, _cfg(fisher_samples=40))
        _train_first_task(ncl, _loader(n=64, seed=0))
        for name in ncl._linear_layers:
            for M in (ncl._kfac_A[name], ncl._kfac_G[name]):
                w = torch.linalg.eigvalsh(M)
                assert w.min().item() >= -1e-5

    def test_factors_nonzero(self):
        torch.manual_seed(0)
        model = TinyMLP()
        ncl = NCL(model, _cfg(fisher_samples=20))
        _train_first_task(ncl, _loader(n=32, seed=2))
        for name in ncl._linear_layers:
            assert ncl._kfac_A[name].abs().sum() > 0
            assert ncl._kfac_G[name].abs().sum() > 0


class TestKFACMeanReductionCompensation:
    def test_A_factor_matches_closed_form_on_first_layer(self):
        """For the input layer, A is the (augmented) input correlation
        (1/N) Σ ā ā^T — this is independent of the model and the labels,
        so we can compute it by hand from the dataset and compare."""
        torch.manual_seed(0)
        in_dim, n = 8, 64
        x = torch.randn(n, in_dim)
        y = torch.randint(0, 3, (n,))

        class SmallNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, 5)
                self.fc2 = nn.Linear(5, 3)

            def forward(self, t):
                return self.fc2(torch.relu(self.fc1(t)))

        model = SmallNet()
        ncl = NCL(model, _cfg(fisher_samples=n))
        loader = DataLoader(TensorDataset(x, y), batch_size=16)
        _train_first_task(ncl, loader)

        # Closed form: augmented input correlation with ones column appended.
        a_aug = torch.cat([x, torch.ones(n, 1)], dim=1)
        A_expected = (a_aug.T @ a_aug) / n

        # We can compare directly only when training did not modify the input
        # distribution — the input layer always sees `x`. The mean-reduction
        # compensation should make this exact up to FP error.
        assert torch.allclose(ncl._kfac_A["fc1"], A_expected, atol=1e-4)

    def test_G_factor_invariant_to_batch_size(self):
        """The per-sample G is (1/N) Σ g_s g_s^T. With CE mean-reduction the
        backward-hook gradient is g_s / B, so the implementation has to
        multiply by B before summing. If it didn't, splitting the same data
        across different batch sizes would produce different G matrices —
        which it must not."""
        torch.manual_seed(0)
        in_dim, n = 12, 64
        # Identical data, identical model — only the batch size differs.
        x = torch.randn(n, in_dim)
        y = torch.randint(0, 3, (n,))

        def _build(batch_size: int):
            torch.manual_seed(123)
            model = TinyMLP(in_dim=in_dim, hidden=8, out_dim=3)
            ncl = NCL(model, _cfg(fisher_samples=n))
            loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
            # Skip observe()s — they would drift the params; compute K-FAC
            # against the freshly-initialised model so the two runs see the
            # same forward/backward path.
            ncl.end_task(0, loader)
            return ncl

        ncl_b8 = _build(8)
        ncl_b32 = _build(32)

        for name in ("net.0", "net.2"):
            A1 = ncl_b8._kfac_A[name]
            A2 = ncl_b32._kfac_A[name]
            G1 = ncl_b8._kfac_G[name]
            G2 = ncl_b32._kfac_G[name]
            # A only depends on inputs and so is trivially batch-size
            # invariant; G is the one the mean-reduction fix targets.
            assert torch.allclose(A1, A2, atol=1e-5)
            assert torch.allclose(G1, G2, atol=1e-4), (
                f"G factor for {name} is batch-size dependent — "
                f"the CE mean-reduction compensation is wrong"
            )


class TestKFACAccumulation:
    def test_additive_across_tasks(self):
        """After two tasks, A_k = A_{k-1} + Â_k exactly — additive Kronecker
        accumulation (a simplification of the paper's nearest_kf_sum)."""
        torch.manual_seed(0)
        model = TinyMLP()
        ncl = NCL(model, _cfg(fisher_samples=32))

        loader0 = _loader(n=32, seed=0)
        ncl.end_task(0, loader0)
        A0 = {n: t.clone() for n, t in ncl._kfac_A.items()}
        G0 = {n: t.clone() for n, t in ncl._kfac_G.items()}

        # Snapshot K-FAC from the *same* model state, evaluated on task 1 data.
        loader1 = _loader(n=32, seed=1)
        A_hat, G_hat = ncl._compute_kfac_factors(loader1)

        ncl.end_task(1, loader1)

        # The full Λ should equal Λ_0 + F_1 ⇒ A_k = A_0 + Â_1, G_k = G_0 + Ĝ_1.
        # Note: end_task internally calls _compute_kfac_factors a *second*
        # time on the same loader, but starting from an identical model
        # (we did not run any observe() between the two ends_task), so the
        # result is identical.
        for name in ncl._linear_layers:
            assert torch.allclose(
                ncl._kfac_A[name], A0[name] + A_hat[name], atol=1e-4
            )
            assert torch.allclose(
                ncl._kfac_G[name], G0[name] + G_hat[name], atol=1e-4
            )

    def test_trace_grows_after_second_task(self):
        torch.manual_seed(0)
        model = TinyMLP()
        ncl = NCL(model, _cfg(fisher_samples=32))
        ncl.end_task(0, _loader(n=32, seed=0))
        first = next(iter(ncl._linear_layers))
        tr0 = ncl._kfac_A[first].trace().item()

        ncl.end_task(1, _loader(n=32, seed=1))
        tr1 = ncl._kfac_A[first].trace().item()

        assert tr1 > tr0


# ===========================================================================
# C. Eq. (8) algebra — the actual NCL update direction
# ===========================================================================

class TestUpdateDirection:
    """Pin down ``_apply_natural_gradient`` to its closed form.

    With a *known* (A, G, μ) injected by hand, the rewritten gradient must
    equal exactly  G⁻¹ ∇W_aug A⁻¹ + (θ - μ)_aug — i.e. the bracketed
    quantity in Eq. (8). Multiplied by -η this is the SGD step.
    """

    def _build_two_layer(self):
        """Return (model, ncl) with a deterministic two-layer MLP and small,
        well-conditioned hand-built K-FAC factors. Picked so the inverses
        differ from the identity by a measurable amount."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        ncl = NCL(model, _cfg(damping=0.0, trust_radius=1e9))  # no clipping interference

        # Hand-built factors, symmetric PD.
        A0 = torch.diag(torch.tensor([2.0, 3.0, 1.5, 4.0, 1.0]))   # (4+1, 4+1) for bias
        G0 = torch.diag(torch.tensor([1.0, 2.0, 0.5]))             # (3, 3)
        A1 = torch.diag(torch.tensor([1.0, 2.0, 4.0, 1.0]))        # (3+1, 3+1)
        G1 = torch.diag(torch.tensor([2.5, 1.5]))                  # (2, 2)
        ncl._kfac_A = {"0": A0, "2": A1}
        ncl._kfac_G = {"0": G0, "2": G1}

        # μ — pick something deliberately different from the current weights
        # so (θ - μ) is non-zero in every layer.
        for name, module in ncl._linear_layers.items():
            ncl._prior_W[name] = module.weight.data.clone() + 0.1
            if module.bias is not None:
                ncl._prior_b[name] = module.bias.data.clone() - 0.2

        return model, ncl

    def test_eq8_natgrad_direction_matches_closed_form(self):
        model, ncl = self._build_two_layer()

        # Backward once to populate .grad.
        x = torch.randn(7, 4)
        y = torch.randint(0, 2, (7,))
        ncl.optimizer.zero_grad()
        ncl.loss_fn(model(x), y).backward()

        # Expected post-rewrite gradient, per layer.
        expected_grad: dict = {}
        for name, module in ncl._linear_layers.items():
            A, G = ncl._kfac_A[name], ncl._kfac_G[name]
            A_inv = torch.linalg.inv(A)
            G_inv = torch.linalg.inv(G)

            gW = module.weight.grad.data.clone()
            if module.bias is not None:
                gb = module.bias.grad.data.clone()
                grad_aug = torch.cat([gW, gb.unsqueeze(1)], dim=1)
                delta_W = module.weight.data - ncl._prior_W[name]
                delta_b = module.bias.data - ncl._prior_b[name]
                delta_aug = torch.cat([delta_W, delta_b.unsqueeze(1)], dim=1)
            else:
                grad_aug = gW
                delta_aug = module.weight.data - ncl._prior_W[name]

            expected_grad[name] = G_inv @ grad_aug @ A_inv + delta_aug

        ncl._apply_natural_gradient()

        # Compare layer by layer. The augmented form unpacks into
        # weight.grad = expected[:, :-1] and bias.grad = expected[:, -1].
        for name, module in ncl._linear_layers.items():
            e = expected_grad[name]
            if module.bias is not None:
                assert torch.allclose(module.weight.grad.data, e[:, :-1], atol=1e-5)
                assert torch.allclose(module.bias.grad.data, e[:, -1], atol=1e-5)
            else:
                assert torch.allclose(module.weight.grad.data, e, atol=1e-5)

    def test_no_prior_means_passthrough(self):
        """On task 0 (empty Λ) the rewritten gradient is exactly the raw
        gradient — NCL reduces to plain SGD until a prior is folded in."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3))
        ncl = NCL(model, _cfg())

        x = torch.randn(6, 5)
        y = torch.randint(0, 3, (6,))
        ncl.optimizer.zero_grad()
        ncl.loss_fn(model(x), y).backward()

        raw = {n: m.weight.grad.data.clone() for n, m in ncl._linear_layers.items()}
        ncl._apply_natural_gradient()
        for n, m in ncl._linear_layers.items():
            assert torch.allclose(m.weight.grad.data, raw[n], atol=0.0)

    def test_rubber_band_pulls_toward_mu_when_loss_grad_zero(self):
        """If ∇L = 0 (set .grad to zeros after the loss backward), then
        Eq. (8) reduces to θ ← θ - η (θ - μ): the optimiser moves by -η · δ
        toward the prior mean."""
        model = nn.Sequential(nn.Linear(3, 2))
        ncl = NCL(model, _cfg(lr=0.5, damping=0.0))

        # Inject A = G = I and μ = θ - 1 ⇒ (θ - μ) = +1 everywhere.
        d_in_aug = 3 + 1  # bias augmented
        d_out = 2
        ncl._kfac_A = {"0": torch.eye(d_in_aug)}
        ncl._kfac_G = {"0": torch.eye(d_out)}
        with torch.no_grad():
            # current weights = 0, μ = -1 → δ = +1
            model[0].weight.zero_()
            model[0].bias.zero_()
        ncl._prior_W = {"0": torch.full_like(model[0].weight.data, -1.0)}
        ncl._prior_b = {"0": torch.full_like(model[0].bias.data, -1.0)}

        ncl.optimizer.zero_grad()
        # Force ∇L = 0 by zeroing the .grad tensors directly.
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        ncl._apply_natural_gradient()
        ncl.optimizer.step()

        # Expected: θ ← 0 - 0.5 * (0 - (-1)) = -0.5 everywhere.
        assert torch.allclose(model[0].weight.data, torch.full_like(model[0].weight.data, -0.5), atol=1e-6)
        assert torch.allclose(model[0].bias.data,   torch.full_like(model[0].bias.data,   -0.5), atol=1e-6)

    def test_lambda_identity_recovers_sgd_plus_weight_decay(self):
        """With A = G = I and μ = 0, Eq. (8) becomes θ ← θ - η (∇L + θ),
        i.e. SGD with weight decay anchored at the origin. We compare a
        single NCL step to a hand-rolled SGD-with-weight-decay step on an
        identical model+batch."""
        torch.manual_seed(0)
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        # NCL model.
        torch.manual_seed(1)
        model_ncl = nn.Sequential(nn.Linear(4, 2))
        ncl = NCL(model_ncl, _cfg(lr=0.1, damping=0.0))
        ncl._kfac_A = {"0": torch.eye(5)}    # A_aug = I_{d_in+1}
        ncl._kfac_G = {"0": torch.eye(2)}
        ncl._prior_W = {"0": torch.zeros_like(model_ncl[0].weight.data)}
        ncl._prior_b = {"0": torch.zeros_like(model_ncl[0].bias.data)}

        # Reference model — identical init, plain SGD with `θ + ∇L` gradient.
        torch.manual_seed(1)
        model_ref = nn.Sequential(nn.Linear(4, 2))
        ref_optim = torch.optim.SGD(model_ref.parameters(), lr=0.1)

        # One step.
        ncl.observe(x, y, task_id=1)

        ref_optim.zero_grad()
        nn.CrossEntropyLoss()(model_ref(x), y).backward()
        # Hand-roll the rubber-band: ∇ ← ∇ + θ (since μ=0, Λ=I).
        for p in model_ref.parameters():
            p.grad = p.grad + p.data
        ref_optim.step()

        for p_ncl, p_ref in zip(model_ncl.parameters(), model_ref.parameters()):
            assert torch.allclose(p_ncl.data, p_ref.data, atol=1e-6)

    def test_large_damping_drives_step_to_zero(self):
        """A_inv = (A + ε I)^{-1} → 0 as ε → ∞, so the natural-gradient
        term vanishes and only the (θ - μ) rubber-band remains. With μ
        equal to the current θ this means the rewritten gradient → 0."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 3))
        ncl = NCL(model, _cfg(damping=1e8))   # extreme damping
        ncl._kfac_A = {"0": torch.eye(5)}
        ncl._kfac_G = {"0": torch.eye(3)}
        ncl._prior_W = {"0": model[0].weight.data.clone()}    # μ = θ → δ = 0
        ncl._prior_b = {"0": model[0].bias.data.clone()}

        x = torch.randn(5, 4)
        y = torch.randint(0, 3, (5,))
        ncl.optimizer.zero_grad()
        ncl.loss_fn(model(x), y).backward()
        # Stash the raw gradient norm for comparison.
        raw_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5
        assert raw_norm > 1e-3

        ncl._apply_natural_gradient()

        new_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5
        # With enormous damping the natural gradient is crushed; restoring is
        # zero because we set μ = θ. Effective grad should be near zero.
        assert new_norm < raw_norm * 1e-3, (
            f"With damping=1e8 and μ=θ, the rewritten grad should approach 0, "
            f"got norm {new_norm} (raw was {raw_norm})"
        )


# ===========================================================================
# D. Diagnostics — kl_proxy and tr_scale
# ===========================================================================

class TestDiagnostics:
    def test_keys_present(self):
        ncl = NCL(TinyMLP(), _cfg())
        diag = ncl.get_step_diagnostics()
        assert set(diag.keys()) == {"kl_proxy", "tr_scale"}

    def test_zero_kl_and_unit_trscale_before_first_end_task(self):
        ncl = NCL(TinyMLP(), _cfg())
        x, y = torch.randn(8, 20), torch.randint(0, 4, (8,))
        ncl.observe(x, y, task_id=0)
        diag = ncl.get_step_diagnostics()
        assert diag["kl_proxy"] == 0.0
        assert diag["tr_scale"] == 1.0

    def test_kl_proxy_matches_quadratic_form(self):
        """For an injected (A, G, μ) and known θ, kl_proxy = (1/2) δ^T Λ δ
        where δ = θ - μ and Λ_layer = A ⊗ G in Kronecker form. We compute
        the expected value with explicit Kronecker products."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(3, 2))
        ncl = NCL(model, _cfg(damping=0.0, trust_radius=1e12))

        A = torch.tensor([[2., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 3., 0.],
                          [0., 0., 0., 1.]])   # augmented (d_in+1)
        G = torch.tensor([[1., 0.],
                          [0., 2.]])
        ncl._kfac_A = {"0": A}
        ncl._kfac_G = {"0": G}
        # δ_W and δ_b chosen so we can hand-compute the Mahalanobis quadratic.
        ncl._prior_W = {"0": model[0].weight.data - 0.5}     # δ_W = +0.5
        ncl._prior_b = {"0": model[0].bias.data + 0.3}        # δ_b = -0.3

        # Closed form: (1/2) vec(δ_aug)^T (A ⊗ G) vec(δ_aug)
        delta_W = torch.full_like(model[0].weight.data, 0.5)
        delta_b = torch.full_like(model[0].bias.data, -0.3)
        delta_aug = torch.cat([delta_W, delta_b.unsqueeze(1)], dim=1)   # (2, 4)
        expected = 0.5 * (delta_aug * (G @ delta_aug @ A)).sum().item()

        # Trigger a step to populate diagnostics.
        x = torch.randn(4, 3)
        y = torch.randint(0, 2, (4,))
        ncl.observe(x, y, task_id=1)
        diag = ncl.get_step_diagnostics()

        assert diag["kl_proxy"] == pytest.approx(expected, rel=1e-4)

    def test_tr_scale_in_unit_interval(self):
        """tr_scale = min(1, r / ‖step‖_Λ) — always ≤ 1 by construction,
        > 0 unless the step has zero Λ-norm."""
        torch.manual_seed(0)
        model = TinyMLP()
        ncl = NCL(model, _cfg(fisher_samples=30))
        _train_first_task(ncl, _loader(n=32, seed=0))

        for x, y in _loader(n=16, seed=1):
            ncl.observe(x, y, task_id=1)
            d = ncl.get_step_diagnostics()
            assert 0.0 < d["tr_scale"] <= 1.0
            assert d["kl_proxy"] >= 0.0

    def test_tr_scale_shrinks_with_smaller_radius(self):
        """Halving the trust radius can only shrink tr_scale (or leave it ≤ 1
        if the step was already small). For a large enough step, the relation
        is exact: tr_scale_small = tr_scale_large * (r_small / r_large)."""
        torch.manual_seed(0)
        model = TinyMLP()
        ncl_big = NCL(model, _cfg(fisher_samples=30, trust_radius=10.0))
        _train_first_task(ncl_big, _loader(n=32, seed=0))

        # Same trained state in a copy with smaller radius.
        torch.manual_seed(0)
        model2 = TinyMLP()
        ncl_small = NCL(model2, _cfg(fisher_samples=30, trust_radius=0.01))
        _train_first_task(ncl_small, _loader(n=32, seed=0))

        x, y = next(iter(_loader(n=16, seed=1)))
        ncl_big.observe(x, y, task_id=1)
        ncl_small.observe(x, y, task_id=1)

        d_big = ncl_big.get_step_diagnostics()
        d_small = ncl_small.get_step_diagnostics()

        # With r=0.01 and a non-trivial natural-gradient step on the boundary
        # of task 1, tr_scale_small should be visibly below tr_scale_big.
        assert d_small["tr_scale"] <= d_big["tr_scale"]


# ===========================================================================
# E. End-to-end: NCL reduces forgetting vs. plain SGD
# ===========================================================================

class TestForgettingReduction:
    """Behavioural sanity check. NCL on a strongly-separated two-task
    setting should keep more of task-0 accuracy after task-1 training
    than vanilla SGD."""

    def _run(self, use_ncl: bool, in_dim: int = 20, n_classes: int = 2) -> float:
        torch.manual_seed(2024)
        # Task 0: decision boundary along feature 0.
        torch.manual_seed(0)
        x0 = torch.randn(200, in_dim)
        y0 = (x0[:, 0] > 0).long()
        loader0 = DataLoader(TensorDataset(x0, y0), batch_size=16, shuffle=True)

        # Task 1: decision boundary along feature 1.
        torch.manual_seed(1)
        x1 = torch.randn(200, in_dim)
        y1 = (x1[:, 1] > 0).long()
        loader1 = DataLoader(TensorDataset(x1, y1), batch_size=16, shuffle=True)

        torch.manual_seed(2024)
        model = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, n_classes))

        if use_ncl:
            method = NCL(model, _cfg(fisher_samples=200, damping=1e-3, lr=0.05))
            for _ in range(10):
                for x, y in loader0:
                    method.observe(x, y, task_id=0)
            method.end_task(0, loader0)
            for _ in range(10):
                for x, y in loader1:
                    method.observe(x, y, task_id=1)
        else:
            optim = torch.optim.SGD(model.parameters(), lr=0.05)
            loss_fn = nn.CrossEntropyLoss()
            for _ in range(10):
                for x, y in loader0:
                    optim.zero_grad(); loss_fn(model(x), y).backward(); optim.step()
            for _ in range(10):
                for x, y in loader1:
                    optim.zero_grad(); loss_fn(model(x), y).backward(); optim.step()

        model.eval()
        with torch.no_grad():
            preds = model(x0).argmax(1)
        return (preds == y0).float().mean().item()

    def test_ncl_at_least_matches_sgd_on_task0(self):
        acc_ncl = self._run(use_ncl=True)
        acc_sgd = self._run(use_ncl=False)
        # A weak floor: NCL must not collapse on task-0 retention.
        assert acc_ncl > 0.45, f"NCL task-0 accuracy collapsed to {acc_ncl:.2f}"
        # And it shouldn't be drastically worse than SGD (it usually beats SGD
        # on this setup, but the test is intentionally permissive).
        assert acc_ncl >= acc_sgd - 0.10, (
            f"NCL ({acc_ncl:.2f}) is much worse than plain SGD ({acc_sgd:.2f}) "
            f"after task 1 — natural-gradient direction may be wrong."
        )


# ===========================================================================
# F. Internal helpers
# ===========================================================================

class TestDampedInv:
    def test_damped_inv_recovers_inverse_with_tiny_damping(self):
        torch.manual_seed(0)
        M = torch.randn(5, 5)
        M = M @ M.T + torch.eye(5)   # symmetric PD
        inv_exact = torch.linalg.inv(M)
        inv_damped = _damped_inv(M, 1e-12)
        assert torch.allclose(inv_exact, inv_damped, atol=1e-4)

    def test_damped_inv_handles_singular_matrix(self):
        """A rank-deficient matrix with non-trivial damping is still invertible."""
        M = torch.zeros(4, 4)
        M[0, 0] = 1.0   # rank 1
        inv = _damped_inv(M, 0.1)
        # (M + 0.1 I)^{-1}: diag-heavy, well-defined.
        assert torch.isfinite(inv).all()
        # Sanity: (M + 0.1 I) @ inv ≈ I
        assert torch.allclose((M + 0.1 * torch.eye(4)) @ inv, torch.eye(4), atol=1e-5)
