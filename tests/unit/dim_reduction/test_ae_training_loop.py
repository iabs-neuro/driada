"""Training-loop behaviour shared by the autoencoder embeddings."""

import numpy as np
import pytest
import torch

from driada.dim_reduction.data import MVData


def _data(n_samples, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.randn(3, n_samples)
    return np.vstack([base, base + 0.3 * rng.randn(3, n_samples)])


def test_ae_returns_the_best_epoch_not_the_last():
    """The model kept at the end must be the one with the lowest held-out loss.

    No dropout anywhere, so the held-out loss of the returned model can be
    recomputed exactly; a learning rate this large makes later epochs worse
    than the best one, so returning the last epoch would not match.
    """
    data = _data(100)
    emb = MVData(data).get_embedding(
        method="ae",
        dim=2,
        epochs=40,
        lr=0.5,
        seed=0,
        batch_size=32,
        feature_dropout=0.0,
        enc_kwargs={"dropout": 0.0},
        dec_kwargs={"dropout": 0.0},
        verbose=False,
    )

    held_out = torch.tensor(data[:, 80:].T).float()
    model = emb.nnmodel
    with torch.no_grad():
        loss = torch.nn.functional.mse_loss(model(held_out), held_out).item()

    assert np.isclose(loss, emb.nn_loss, rtol=1e-5)


@pytest.mark.parametrize("flag", ["add_mi_loss", "add_corr_loss"])
def test_ae_validation_with_extra_losses_and_uneven_batches(flag):
    """Validation must use the codes of its own batch, also when the last
    training batch has a different size from the validation batches."""
    data = _data(50)
    extra = {"add_mi_loss": {"mi_hyperweight": 0.1, "minimize_mi_data": data[:1]},
             "add_corr_loss": {"corr_hyperweight": 0.1}}[flag]
    emb = MVData(data).get_embedding(
        method="ae",
        dim=2,
        epochs=2,
        seed=0,
        batch_size=16,
        verbose=False,
        **{flag: True},
        **extra,
    )
    assert emb.coords.shape == (2, 50)
