"""测试公共件：用例 fixture + 未实现算子的 xfail 包装。

测试结果三态是有意的：
  xfailed  函数体还是 NotImplementedError（没开始写）
  failed   写了但和基准对不上
  passed   对上了
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import cases  # noqa: E402

GRAY_NAMES = list(cases.GRAY_CASES)
BINARY_NAMES = list(cases.BINARY_CASES)


def call(fn, *args, **kwargs):
    """调用算子；若仍是 NotImplementedError 则把该测试标记为 xfail 而不是 failed。"""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{getattr(fn, '__name__', fn)} 尚未实现")


@pytest.fixture(params=GRAY_NAMES)
def gray(request) -> np.ndarray:
    return cases.make(request.param)


@pytest.fixture(params=BINARY_NAMES)
def binary(request) -> np.ndarray:
    return cases.make(request.param)


def assert_close(a: np.ndarray, b: np.ndarray, tol: float, what: str) -> None:
    assert a.shape == b.shape, f"{what}: shape 不一致 {a.shape} vs {b.shape}"
    d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    assert d <= tol, f"{what}: max|err| = {d:.3e} 超过容差 {tol:.1e}"
