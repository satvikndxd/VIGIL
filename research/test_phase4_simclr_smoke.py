"""Smoke test for the proposed VIGIL Phase 4 SimCLR module.

This does not train on, download, or fabricate ECG data. It validates tensor
contracts with zero-valued placeholder waveforms only.
"""

import torch

from phase4_simclr_ecg import ECGClassifier, ECGSimCLR, ECGSimCLRAugment, NTXentLoss, set_global_seed


if __name__ == "__main__":
    set_global_seed(42)
    batch = torch.zeros(8, 180)

    augment = ECGSimCLRAugment()
    view_one = augment(batch)
    view_two = augment(batch)
    assert view_one.shape == (8, 180)
    assert view_two.shape == (8, 180)

    simclr = ECGSimCLR(in_channels=1, embedding_dim=64, projection_dim=32)
    z_one = simclr(view_one)
    z_two = simclr(view_two)
    assert z_one.shape == (8, 32)
    assert z_two.shape == (8, 32)

    loss = NTXentLoss(temperature=0.10)(z_one, z_two)
    assert torch.isfinite(loss)

    classifier = ECGClassifier(simclr.encoder, num_classes=5)
    logits = classifier(batch)
    assert logits.shape == (8, 5)

    print(f"smoke_test_passed=true ntxent={loss.item():.6f} logits_shape={tuple(logits.shape)}")
