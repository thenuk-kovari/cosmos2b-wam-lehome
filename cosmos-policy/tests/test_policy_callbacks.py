import pytest
import torch

from cosmos_policy.config import callbacks


def test_sparse_metric_rank_with_no_local_samples_still_reduces(monkeypatch):
    reductions = []

    def fake_all_reduce(stats, op):
        reductions.append((stats.clone(), op))
        # Simulate two samples with a summed loss of six on other ranks.
        stats.add_(stats.new_tensor([6.0, 2.0]))

    monkeypatch.setattr(callbacks.dist, "all_reduce", fake_all_reduce)

    record = callbacks._LossRecordNoEDM()
    average = record.get_stat()

    assert average == pytest.approx(3.0)
    assert len(reductions) == 1
    torch.testing.assert_close(reductions[0][0].cpu(), torch.tensor([0.0, 0.0]))
    assert reductions[0][1] == callbacks.dist.ReduceOp.SUM
    assert record.loss == 0
    assert record.iter_count == 0
