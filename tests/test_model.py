"""Unit tests for src/model.py: forward-pass shapes, gradient flow, and
the negative-Pearson loss."""

import torch

from src.model import NegativePearsonLoss, PhysNetLite


class TestPhysNetLite:
    def test_forward_output_shape(self):
        model = PhysNetLite(base_channels=8)
        x = torch.rand(2, 3, 32, 36, 36)  # small clip for speed
        out = model(x)
        assert out.shape == (2, 32)

    def test_output_is_zscored_per_sample(self):
        model = PhysNetLite(base_channels=8)
        model.eval()
        x = torch.rand(3, 3, 32, 36, 36)
        with torch.no_grad():
            out = model(x)
        assert torch.allclose(out.mean(dim=1), torch.zeros(3), atol=1e-4)
        assert torch.allclose(out.std(dim=1), torch.ones(3), atol=1e-2)

    def test_gradients_flow_to_all_params(self):
        model = PhysNetLite(base_channels=8)
        x = torch.rand(2, 3, 16, 32, 32)
        target = torch.randn(2, 16)
        loss_fn = NegativePearsonLoss()

        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()

        n_missing_grad = 0
        for name, p in model.named_parameters():
            if p.grad is None or torch.all(p.grad == 0):
                n_missing_grad += 1
        assert n_missing_grad == 0, "Some parameters received no gradient"

    def test_parameter_count_is_small(self):
        """Constraint from the spec: must train on a single consumer
        GPU or CPU -- sanity-check the model stays compact."""
        model = PhysNetLite(base_channels=16)
        n_params = model.count_parameters()
        assert n_params < 2_000_000, f"Model has {n_params} params, expected <2M"

    def test_handles_variable_clip_length(self):
        model = PhysNetLite(base_channels=8)
        for t in (16, 32, 64):
            x = torch.rand(1, 3, t, 36, 36)
            out = model(x)
            assert out.shape == (1, t)


class TestNegativePearsonLoss:
    def test_zero_loss_for_identical_signals(self):
        loss_fn = NegativePearsonLoss()
        signal = torch.randn(4, 50)
        loss = loss_fn(signal, signal.clone())
        assert loss.item() < 1e-5

    def test_loss_near_two_for_anticorrelated_signals(self):
        loss_fn = NegativePearsonLoss()
        signal = torch.randn(4, 50)
        loss = loss_fn(signal, -signal)
        assert loss.item() > 1.9

    def test_loss_invariant_to_scale_and_shift(self):
        """Pearson correlation loss should be unaffected by an affine
        transform of the prediction -- this is the whole point of using
        it instead of MSE, since we don't know the model's output scale
        relative to the ground-truth waveform's units."""
        loss_fn = NegativePearsonLoss()
        torch.manual_seed(0)
        pred = torch.randn(2, 40)
        target = torch.randn(2, 40)
        loss_a = loss_fn(pred, target)
        loss_b = loss_fn(pred * 5.0 + 3.0, target)
        assert torch.allclose(loss_a, loss_b, atol=1e-5)
