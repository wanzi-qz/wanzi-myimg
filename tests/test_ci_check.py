"""CI 门禁（tools/ci_check.py）自己的回归。

门禁脚本最尴尬的失败方式是它自己坏了却一路 exit 0，CI 天天绿、其实什么都没查。
所以这里把它的每条判定都钉一遍：绿报告必须过，真差异必须拦，基线漂了必须拦，
组数不够必须拦。顺带钉一条项目纪律 —— 「仅记录」是口径差异，不许被当成红牌。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# tools/ 不是包（没有 __init__.py），按路径加载
_spec = importlib.util.spec_from_file_location("ci_check", str(ROOT / "tools" / "ci_check.py"))
CC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CC)

PINS = (
    "# 注释行要被跳过\n"
    "numpy==1.26.4\n"
    "opencv-python==4.10.0.84   # 行尾注释也要能剥掉\n"
    "scikit-image==0.25.2\n"
    "\n"
    "matplotlib>=3.7\n"          # 没钉死的（>=）不进字典
)
ENV_OK = {"python": "3.12.3", "numpy": "1.26.4", "opencv": "4.10.0", "skimage": "0.25.2"}


def _row(op="gaussian_filter", case="microscope", verdict="逐像素精确", **kw):
    r = {"op": op, "case": case, "verdict": verdict, "exact": True, "max_abs": 0.0, "note": ""}
    r.update(kw)
    return r


def _green_rows(n=None):
    """凑够 MIN_ROWS 条绿行（少一条都不行，组数本身就是一道门）。"""
    n = CC.MIN_ROWS if n is None else n
    return [_row(case=f"c{i}") for i in range(n)]


def _write(tmp_path, rows, env=None, pins=PINS):
    js = tmp_path / "compare.json"
    js.write_text(json.dumps({"generated_at": "2026-10-21 00:00:00",
                              "env": ENV_OK if env is None else env,
                              "rows": rows}, ensure_ascii=False), encoding="utf-8")
    rq = tmp_path / "reqs.txt"
    rq.write_text(pins, encoding="utf-8")
    return js, rq


def _run(js, rq, *extra):
    return CC.main([str(js), "--reqs", str(rq), *extra])


# ------------------------------------------------------------------ parse_pins

def test_parse_pins_skips_comments_and_loose_specs(tmp_path):
    p = tmp_path / "reqs.txt"
    p.write_text(PINS, encoding="utf-8")
    pins = CC.parse_pins(p)
    assert pins == {"numpy": "1.26.4", "opencv-python": "4.10.0.84", "scikit-image": "0.25.2"}
    assert "matplotlib" not in pins      # >= 不算钉子


# ------------------------------------------------------------------ 绿 / 红

def test_green_report_passes(tmp_path, capsys):
    js, rq = _write(tmp_path, _green_rows())
    assert _run(js, rq) == 0
    out = capsys.readouterr().out
    assert "✅" in out
    assert f"{CC.MIN_ROWS} 组" in out


def test_bad_verdict_fails_and_names_the_operator(tmp_path, capsys):
    rows = _green_rows()
    rows[3] = _row(op="median_filter", case="impulse", verdict="超出容差",
                   exact=False, max_abs=12.0, note="边界 REPLICATE 没对齐")
    js, rq = _write(tmp_path, rows)
    assert _run(js, rq) == 1
    out = capsys.readouterr().out
    assert "❌" in out
    assert "median_filter / impulse" in out          # 得指到具体算子和用例
    assert "1.200e+01" in out                        # 误差数字要原样打出来
    assert "边界 REPLICATE 没对齐" in out             # note 带上，省得再去翻报告


@pytest.mark.parametrize("verdict", CC.BAD_VERDICTS)
def test_all_bad_verdicts_are_caught(tmp_path, verdict):
    rows = _green_rows()
    rows[0] = _row(verdict=verdict, exact=False)
    js, rq = _write(tmp_path, rows)
    assert _run(js, rq) == 1


def test_record_only_is_not_a_failure(tmp_path):
    """口径差异（clahe / canny_smoothed）标 🟡 只记录，不许被门禁当红牌。

    这条是项目纪律：把它们判成失败，就等于逼着人去「修」一个本来就不是 bug 的东西。
    """
    rows = _green_rows()
    rows[0] = _row(op="clahe", verdict="仅记录", exact=False, max_abs=35.0)
    rows[1] = _row(op="canny_smoothed", verdict="仅记录", exact=False, iou=0.1)
    js, rq = _write(tmp_path, rows)
    assert _run(js, rq) == 0


def test_too_few_rows_fails(tmp_path, capsys):
    """组数不够 = 有用例被静默跳过（多半是 --op 名字打错），必须拦。"""
    js, rq = _write(tmp_path, _green_rows(3))
    assert _run(js, rq) == 1
    assert "静默跳过" in capsys.readouterr().out


def test_empty_report_fails(tmp_path):
    js, rq = _write(tmp_path, [])
    assert _run(js, rq) == 1


# ------------------------------------------------------------------ 基线核对

def test_numpy_drift_fails(tmp_path, capsys):
    env = dict(ENV_OK, numpy="2.1.0")
    js, rq = _write(tmp_path, _green_rows(), env=env)
    assert _run(js, rq) == 1
    assert "numpy" in capsys.readouterr().out


def test_opencv_four_segment_pin_matches_three_segment_env(tmp_path):
    """pip 报 4.10.0.84、cv2.__version__ 报 4.10.0，是同一个东西，不许误报。"""
    assert _run(*_write(tmp_path, _green_rows())) == 0


def test_unpinned_baseline_package_fails(tmp_path, capsys):
    """钉子里少了 scikit-image（或写成 >=），门禁要报出来而不是装作没看见。"""
    js, rq = _write(tmp_path, _green_rows(), pins="numpy==1.26.4\nopencv-python==4.10.0.84\n")
    assert _run(js, rq) == 1
    assert "scikit-image" in capsys.readouterr().out


def test_missing_reqs_file_fails(tmp_path, capsys):
    js = tmp_path / "compare.json"
    js.write_text(json.dumps({"env": ENV_OK, "rows": _green_rows()}), encoding="utf-8")
    assert CC.main([str(js), "--reqs", str(tmp_path / "nope.txt")]) == 1
    assert "不存在" in capsys.readouterr().out


def test_missing_report_fails(tmp_path, capsys):
    assert CC.main([str(tmp_path / "nope.json"), "--reqs", str(tmp_path / "r.txt")]) == 1
    out = capsys.readouterr().out
    assert "run_compare" in out          # 得告诉人下一步该跑什么


# ------------------------------------------------------------------ python 版本

def test_expect_python_accepts_patch_level(tmp_path):
    js, rq = _write(tmp_path, _green_rows(), env=dict(ENV_OK, python="3.12.7"))
    assert _run(js, rq, "--expect-python", "3.12") == 0


def test_expect_python_rejects_other_minor(tmp_path, capsys):
    js, rq = _write(tmp_path, _green_rows(), env=dict(ENV_OK, python="3.11.9"))
    assert _run(js, rq, "--expect-python", "3.12") == 1
    assert "python" in capsys.readouterr().out


def test_no_expect_python_skips_that_check(tmp_path):
    js, rq = _write(tmp_path, _green_rows(), env=dict(ENV_OK, python="3.9.13"))
    assert _run(js, rq) == 0


# ------------------------------------------------------------------ summarize

def test_summarize_lists_known_verdicts_in_order():
    rows = [_row(verdict="仅记录"), _row(verdict="逐像素精确"), _row(verdict="逐像素精确"),
            _row(verdict="某个新措辞")]
    s = CC.summarize(rows)
    assert s.startswith("4 组：")
    assert s.index("逐像素精确") < s.index("仅记录") < s.index("某个新措辞")


# ------------------------------------------------------------------ 真产物

@pytest.mark.skipif(not (ROOT / "benchmarks" / "out" / "compare.json").exists(),
                    reason="仓库里没有 benchmarks/out/compare.json")
def test_checked_in_report_still_passes():
    """入库那份对拍报告必须过门禁：它是 README / PROGRESS / user_guide 引用的事实源。"""
    assert CC.main([str(ROOT / "benchmarks" / "out" / "compare.json"),
                    "--reqs", str(ROOT / "requirements-ci.txt")]) == 0