import torch

from distributions import TwoPoint, expectation, variance


def test_single_dimensional():
    d = TwoPoint(5, 20, 0.2)
    samples = d.sample((10000,))
    assert expectation(d) == 8
    assert variance(d) == 36
    assert torch.isclose(expectation(d), samples.mean(), rtol=0.02)
    assert torch.isclose(variance(d), samples.var(), rtol=0.02)


def test_multi_dimensional():
    lows = torch.FloatTensor([5, 0])
    highs = torch.FloatTensor([20, 10])
    probs = torch.FloatTensor([0.2, 1.0])
    d = TwoPoint(lows, highs, probs)
    samples = d.sample((10000,))

    assert (expectation(d) == torch.FloatTensor([8, 10])).all()
    assert (variance(d) == torch.FloatTensor([36, 0])).all()
    assert torch.isclose(expectation(d), samples.mean(dim=0), rtol=0.02).all()
    assert torch.isclose(variance(d), samples.var(dim=0), rtol=0.02).all()
