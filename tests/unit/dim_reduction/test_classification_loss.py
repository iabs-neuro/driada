"""Integration tests for ClassificationLoss with ModularAutoencoder."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from driada.dim_reduction.losses import ClassificationLoss
from driada.dim_reduction.flexible_ae import ModularAutoencoder


class TestClassificationLossStandalone:
    def test_positive_loss_with_labels(self):
        loss = ClassificationLoss(num_classes=5, code_dim=8)
        code = torch.randn(32, 8)
        labels = torch.randint(0, 5, (32,))
        val = loss.compute(code, code, code, labels=labels)
        assert val.ndim == 0 and val.item() > 0

    def test_zero_loss_without_labels(self):
        loss = ClassificationLoss(num_classes=5, code_dim=8)
        code = torch.randn(32, 8)
        val = loss.compute(code, code, code)
        assert val.item() == 0.0

    def test_parameters_accessible(self):
        loss = ClassificationLoss(num_classes=5, code_dim=8)
        params = list(loss.parameters())
        assert len(params) == 2  # weight and bias


class TestClassificationLossInModel:
    def test_model_with_classification_loss(self):
        model = ModularAutoencoder(
            input_dim=64, latent_dim=16, hidden_dim=32,
            loss_components=[
                {"name": "reconstruction", "weight": 1.0},
                {"name": "classification", "weight": 0.5, "num_classes": 5},
            ],
        )
        x = torch.randn(32, 64)
        labels = torch.randint(0, 5, (32,))
        total_loss, loss_dict = model.compute_loss(x, labels=labels)
        assert total_loss.item() > 0
        assert any("ClassificationLoss" in k for k in loss_dict)

    def test_model_without_labels_still_works(self):
        model = ModularAutoencoder(
            input_dim=64, latent_dim=16, hidden_dim=32,
            loss_components=[
                {"name": "reconstruction", "weight": 1.0},
                {"name": "classification", "weight": 0.5, "num_classes": 5},
            ],
        )
        x = torch.randn(32, 64)
        total_loss, loss_dict = model.compute_loss(x)
        assert total_loss.item() > 0

    def test_classifier_learns(self):
        """Classification loss should decrease over a few gradient steps."""
        torch.manual_seed(42)
        model = ModularAutoencoder(
            input_dim=32, latent_dim=8, hidden_dim=16,
            loss_components=[
                {"name": "reconstruction", "weight": 0.0},
                {"name": "classification", "weight": 1.0, "num_classes": 3},
            ],
        )
        # Collect all trainable params including loss component params
        all_params = list(model.parameters())
        for lc in model.losses:
            if hasattr(lc, 'parameters'):
                all_params.extend(lc.parameters())
        optimizer = torch.optim.Adam(all_params, lr=1e-3)

        x = torch.randn(64, 32)
        labels = torch.randint(0, 3, (64,))

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            total_loss, _ = model.compute_loss(x, labels=labels)
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        assert losses[-1] < losses[0], "Classification loss should decrease"
