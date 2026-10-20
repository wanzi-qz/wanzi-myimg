"""CI 门禁：把对拍产物读回来，确认「判定没漂、基线没换」。

为什么要有这么个脚本（run_compare 自己 exit 0）：
  * run_compare 是报告工具不是裁判。出现「超出容差」它照样打印一行 [待查] 然后
    正常退出 —— 在终端里人会看见，在 CI 里没人看见，等于没跑。红牌得有人举。
  * 判定结论强依赖上游库的实现细节。numpy 从 1.26 升到 2.x，浮点结合序一变，
    「浮点噪声级一致」可能就成「超出容差」，而这跟代码对错没关系。compare.json 里
    存了 env，正好拿来和 requirements-ci.txt 的钉子逐个对，堵住「库悄悄升级、
    CI 还是绿的」这条路。
  * 组数也是一道门。op / case 名字打错，run_compare 会安静地跑出 0 组然后 exit 0，
    看着像全绿。少于 MIN_ROWS 就报错。

用法（CI 里就是这两行）：
    python benchmarks/run_compare.py --all
    python tools/ci_check.py --expect-python 3.12

本机自查：
    python tools/ci_check.py                        # 读默认的 benchmarks/out/compare.json
    python tools/ci_check.py some/compare.json      # 读指定产物

退出码：0 全绿 / 1 有问题（CI 靠这个失败）。tests/test_ci_check.py 守着这个脚本本身。
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "benchmarks" / "out" / "compare.json"
DEFAULT_REQS = ROOT / "requirements-ci.txt"

# 和 run_compare.main() 里那份「待查」清单同源：这三类是真差异，其余（逐像素精确 /
# 浮点噪声级一致 / 一致 / IoU 合格 / 两边都无边缘 / 仅记录）都不算失败。
BAD_VERDICTS = ("超出容差", "不一致", "IoU 偏低")

# 满配是 112 组（15 算子 / 17 口径 × 12 用例里有效的组合）。低于这个数说明有
# 用例被静默跳过了 —— 通常是 --op / --case 名字打错，run_compare 不会报错。
MIN_ROWS = 100

# compare.json 的 env 里 opencv 报的是 cv2.__version__（三段），pip 的包版本是四段
# （4.10.0.84），说的是同一个东西，所以统一取前三段比。
PIN_TO_ENV = {"numpy": "numpy", "opencv-python": "opencv", "scikit-image": "skimage"}

VERDICT_ORDER = ("逐像素精确", "浮点噪声级一致", "一致", "IoU 合格", "两边都无边缘", "仅记录")


def parse_pins(reqs: pathlib.Path) -> dict[str, str]:
    """读 requirements-ci.txt 的钉子 -> {包名(小写): 版本}。

    整行注释、空行、行尾注释都跳过；只认 `==`（这份文件的纪律就是全钉死，
    出现 >= 说明有人手松了，那就不进字典，后面 check_env 会报「没钉版本」）。
    """
    pins: dict[str, str] = {}
    for raw in reqs.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, ver = line.split("==", 1)
        pins[name.strip().lower()] = ver.strip()
    return pins


def check_env(env: dict, pins: dict[str, str], expect_python: str | None) -> list[str]:
    """对拍基线三连（numpy / opencv / scikit-image）+ Python 次版本，逐个和钉子比。"""
    bad: list[str] = []
    for pkg, key in PIN_TO_ENV.items():
        want = pins.get(pkg)
        got = str(env.get(key, "?"))
        if want is None:
            bad.append(f"{pkg}: requirements-ci.txt 里没钉版本（必须是 ==）")
        elif ".".join(want.split(".")[:3]) != got:
            bad.append(f"{pkg}: 基线钉的是 {want}，实际跑出来 {got}")
    if expect_python:
        got_py = str(env.get("python", "?"))
        if got_py != expect_python and not got_py.startswith(expect_python + "."):
            bad.append(f"python: 期望 {expect_python}.x，实际 {got_py}")
    return bad


def check_rows(rows: list[dict]) -> list[str]:
    """组数够不够 + 有没有真差异。"""
    bad: list[str] = []
    if len(rows) < MIN_ROWS:
        bad.append(f"只有 {len(rows)} 组对拍（期望 >= {MIN_ROWS}），有用例被静默跳过了")
    for r in rows:
        verdict = r.get("verdict")
        if verdict not in BAD_VERDICTS:
            continue
        where = f"{r.get('op')} / {r.get('case')}"
        nums = []
        if isinstance(r.get("max_abs"), (int, float)):
            nums.append(f"max|err|={float(r['max_abs']):.3e}")
        if isinstance(r.get("iou"), (int, float)):
            nums.append(f"IoU={float(r['iou']):.3f}")
        note = r.get("note") or ""
        bad.append(f"{where}: {verdict} {' '.join(nums)}".strip() + (f" | {note}" if note else ""))
    return bad


def summarize(rows: list[dict]) -> str:
    """一行看懂这份报告：112 组：逐像素精确 60，浮点噪声级一致 20，……"""
    counts = collections.Counter(r.get("verdict", "?") for r in rows)
    parts = [f"{k} {counts[k]}" for k in VERDICT_ORDER if counts.get(k)]
    parts += [f"{k} {v}" for k, v in counts.items() if k not in VERDICT_ORDER]
    return f"{len(rows)} 组：" + "，".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CI 门禁：对拍产物 + 依赖基线检查")
    ap.add_argument("report", nargs="?", type=pathlib.Path, default=DEFAULT_JSON,
                    help=f"compare.json 路径，默认 {DEFAULT_JSON.relative_to(ROOT)}")
    ap.add_argument("--reqs", type=pathlib.Path, default=DEFAULT_REQS,
                    help="版本钉子文件，默认 requirements-ci.txt")
    ap.add_argument("--expect-python", default=None,
                    help="期望的 Python 次版本（如 3.12），要和 workflow 里 setup-python 一致；不给就不查")
    args = ap.parse_args(argv)

    if not args.report.exists():
        print(f"❌ 找不到 {args.report}")
        print("   先跑：python benchmarks/run_compare.py --all")
        return 1

    data = json.loads(args.report.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    env = data.get("env", {})
    print(f"报告 {args.report.name} @ {data.get('generated_at', '?')}")
    print(f"环境 python {env.get('python', '?')} / numpy {env.get('numpy', '?')} / "
          f"cv2 {env.get('opencv', '?')} / skimage {env.get('skimage', '?')}")
    print(summarize(rows))

    problems: list[str] = []
    if not args.reqs.exists():
        problems.append(f"{args.reqs.name} 不存在，基线没法核对")
    else:
        pins = parse_pins(args.reqs)
        problems += check_env(env, pins, args.expect_python)
    problems += check_rows(rows)

    if problems:
        print(f"❌ {len(problems)} 条不通过：")
        for p in problems:
            print(f"   - {p}")
        return 1

    print("✅ 判定全绿，依赖基线与 requirements-ci.txt 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())