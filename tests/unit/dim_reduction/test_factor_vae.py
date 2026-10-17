"""Tests for the Factor-VAE loss and its discriminator training."""

import pytest
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from driada.dim_reduction.losses import FactorVAELoss

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")


def _dependent_code(n, latent_dim, generator):
    """Code whose axes are noisy copies of one source: high total correlation."""
    source = torch.randn(n, 1, generator=generator)
    return source + 0.1 * torch.randn(n, latent_dim, generator=generator)


def _shuffle_axes(code, generator):
    """Same code with every axis permuted independently across samples."""
    columns = [code[torch.randperm(code.shape[0], generator=generator), j]
               for j in range(code.shape[1])]
    return torch.stack(columns, dim=1)


class TestFactorVAEPermutation:
    def test_permutation_breaks_dependence_and_keeps_marginals(self):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        code = _dependent_code(1000, 4, g)
        permuted = FactorVAELoss._permute_dims(code)

        # Every axis keeps its own values (marginal distribution)
        assert torch.equal(torch.sort(permuted, dim=0).values,
                           torch.sort(code, dim=0).values)
        # The dependence between axes is destroyed
        corr = np.corrcoef(permuted.numpy().T)
        off_diag = corr[~np.eye(4, dtype=bool)]
        assert np.abs(off_diag).max() < 0.15
        assert np.corrcoef(code.numpy().T)[0, 1] > 0.9


class TestFactorVAEDiscriminator:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_tc_estimate_separates_dependent_from_independent(self, seed):
        latent_dim = 4
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        loss = FactorVAELoss(gamma=1.0, code_dim=latent_dim, discriminator_dims=[64, 64],
                             discriminator_lr=1e-3)

        for _ in range(200):
            code = _dependent_code(128, latent_dim, g)
            loss.compute(code, None, None)
            loss.discriminator_step()

        code = _dependent_code(2000, latent_dim, g)
        with torch.no_grad():
            tc_dependent = loss.compute(code, None, None).item()
            tc_independent = loss.compute(_shuffle_axes(code, g), None, None).item()

        # The true total correlation of these axes is several nats and zero
        # after shuffling; the discriminator estimate must reflect that gap.
        assert tc_dependent > 1.0
        assert tc_independent < 0.0
        assert tc_dependent - tc_independent > 2.0

    def test_optimizers_are_disjoint(self):
        from driada.dim_reduction.flexible_ae import FlexibleVAE

        model = FlexibleVAE(
            input_dim=20, latent_dim=3, hidden_dim=16,
            loss_components=[
                {"name": "reconstruction", "weight": 1.0},
                {"name": "beta_vae", "weight": 1.0, "beta": 1.0},
                {"name": "factor_vae", "weight": 1.0, "gamma": 5.0},
            ],
            device=torch.device("cpu"),
        )
        loss = model.losses[2]
        disc_params = {id(p) for p in loss.discriminator.parameters()}
        disc_opt_params = {id(p) for group in loss.discriminator_optimizer.param_groups
                           for p in group["params"]}
        model_params = {id(p) for p in model.parameters()}

        assert len(disc_params) > 0
        assert disc_opt_params == disc_params
        assert disc_params.isdisjoint(model_params)
        # The discriminator input size is the latent size of the model
        assert loss.discriminator[0].in_features == 3

    def test_discriminator_trained_in_embedding_loop(self, monkeypatch):
        from driada.dim_reduction.data import MVData

        initial = {}
        original_build = FactorVAELoss._build_discriminator

        def recording_build(self, latent_dim):
            original_build(self, latent_dim)
            initial["params"] = [p.detach().clone()
                                 for p in self.discriminator.parameters()]

        monkeypatch.setattr(FactorVAELoss, "_build_discriminator", recording_build)

        rng = np.random.default_rng(0)
        data = rng.random((20, 200))
        emb = MVData(data).get_embedding(
            method="flexible_ae",
            architecture="vae",
            dim=3,
            epochs=3,
            batch_size=32,
            inter_dim=16,
            verbose=False,
            loss_components=[
                {"name": "reconstruction", "weight": 1.0},
                {"name": "beta_vae", "weight": 1.0, "beta": 1.0},
                {"name": "factor_vae", "weight": 1.0, "gamma": 5.0},
            ],
        )

        assert emb.coords.shape == (3, 200)
        assert np.all(np.isfinite(emb.coords))
        loss = emb.nnmodel.losses[2]
        final = [p.detach() for p in loss.discriminator.parameters()]
        assert len(initial["params"]) == len(final)
        assert any(not torch.equal(a, b) for a, b in zip(initial["params"], final))
