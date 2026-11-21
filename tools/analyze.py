#!/usr/bin/env python3

"""결과 폴더 -> 결과 표(stdout), Optrace 타임라인 그림

  python3 tools/analyze.py <결과 폴더> [--figure PATH] [--overwrite]

결과 폴더 구성 (run.sh가 만든 형식, results/week_1도 같다)
  latency/<v>.csv            trial,kind,us   kind: exec(graphExecute), htp(Accelerator execute) 등. 워밍업은 이미 제외
  optrace/<v>/runtrace.json  Optrace 21회 실행 각각의 Inference cycle (첫 회는 워밍업으로 제외)
  optrace/<v>/qhas.json      Optrace 첫 실행의 HTP op별 시작 cycle, 길이, 스레드(tid), dominant path cycle
  outputs/                   (선택) 보드 출력과 FP32 기준값. 있으면 SQNR을 계산한다

입력 파일은 읽기만 한다. 그림은 --figure(기본: <결과 폴더>/figures/optrace_timeline.svg)에 쓰되,
이미 파일이 있으면 --overwrite 없이는 쓰지 않는다 (기존 결과 보존).
"""
# 파일 전체 역할
# 1. 그래프 3개의 측정 CSV, Optrace runtrace, QHAS를 읽고 형식 검사
# 2. graphExecute 중앙값(trial 범위), HTP 실행 중앙값, Optrace cycle 중앙값, HTP op 수, SQNR을 표로 출력
# 3. QHAS의 HTP op 목록으로 스레드별 타임라인 그림(SVG) 생성

import argparse
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 그림
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["v0_naive", "v1_conv1x1", "v2_nhwc"]

# 타임라인 색 분류 (순서가 category()의 반환 번호)
# 계산 자원(HMX, HVX)과 데이터 이동(DMA), 순서 맞추기(대기, 동기화)를 다른 색으로 구분
CATS = [("HMX 행렬 연산 (Conv)", "#2a78d6"), ("HVX 배치 변환 (Transpose 등)", "#eb6834"),
        ("HVX 기타 (Add, 출력 쓰기)", "#1baf7a"), ("DMA 전송 (입출력, 가중치)", "#eda100"),
        ("DMA 완료 대기·설정", "#e87ba4"), ("동기화 (SyncOp 등)", "#008300")]

# qhas.json의 op 항목에 반드시 있어야 하는 필드 (SDK 버전이 달라 형식이 바뀌면 여기서 멈춤)
QHAS_FIELDS = ("htp_op", "tid", "start_cycle", "cycles", "num_dominant_path_cycles",
               "hmx", "hvx", "dma", "dma_wait", "dma_set", "sync")


def die(msg):
    sys.exit(f"[analyze] {msg}")   # 메시지를 stderr에 출력하고 종료 코드 1


def category(r):
    # QHAS가 op마다 표시한 자원 플래그(hmx, hvx, dma, dma_wait, dma_set, sync)로 색 분류
    # op 이름만으로 나누면 v1의 출력 쓰기(HVX가 직접 함)처럼 같은 이름이 다른 자원에서 도는 경우를 틀리게 칠함
    if r["hmx"]:
        return 0                                                                                       # HMX 행렬 연산
    if r["hvx"]:
        layout = ("Transpose", "ForceFormat", "SlicePad", "Concat", "Reshape")
        return 1 if any(k in r["htp_op"] for k in layout) else 2                                      # HVX 배치 변환 / 기타
    if r["dma"]:
        return 3                                                                                       # InputSlice, weights_to_vtcm 등
    if r["dma_wait"] or r["dma_set"]:
        return 4                                                                                       # DmaCheckpointWait/Set
    if r["sync"]:
        return 5                                                                                       # SyncOp, SystemService
    die(f"분류할 수 없는 HTP op: {r['htp_op']}")                                                         # 새 종류가 나오면 멈춤


def read_latency(path):
    # latency CSV(trial,kind,us)를 검사하며 읽고, trial별 exec 중앙값 목록과 htp 값 목록 반환
    if not path.is_file():
        die(f"없음: {path}")
    exec_by_trial, htp = {}, []
    for i, row in enumerate(csv.reader(path.open()), 1):
        if len(row) != 3:                                      # 형식 확인
            die(f"{path}:{i} 열 수가 3이 아님: {row}")
        trial, kind, val = row
        try:
            x = float(val)                                     # 값이 숫자인지 확인
        except ValueError:
            die(f"{path}:{i} 숫자가 아님: {val}")
        if kind == "exec":
            exec_by_trial.setdefault(trial, []).append(x)      # trial별로 모음
        elif kind == "htp":
            htp.append(x)
        # htp_rpc, host_rpc, qnn_execute는 표에 쓰지 않음
    if not exec_by_trial or not htp:
        die(f"{path}: exec 또는 htp 행이 없음")
    if len({len(x) for x in exec_by_trial.values()}) != 1:     # trial마다 측정 행 수가 같아야 함
        die(f"{path}: trial마다 exec 행 수가 다름")
    return [st.median(x) for x in exec_by_trial.values()], htp


def read_json(path):
    # JSON 읽기, 없거나 깨졌으면 중단
    if not path.is_file():
        die(f"없음: {path}")
    txt = path.read_text()
    if path.name == "runtrace.json":
        txt = re.sub(r",\s*\]", "]", txt)  # Reader가 남긴 배열 끝 쉼표 정리 (JSON 규칙 위반)
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        die(f"{path}: JSON 오류 {e}")


def load(res, v):
    # 그래프 하나의 측정값과 QHAS 데이터를 모음
    trial, htp = read_latency(res / f"latency/{v}.csv")
    runs = [e["dur"] for e in read_json(res / f"optrace/{v}/runtrace.json")["traceEvents"] if e.get("name") == "Inference"]   # 21회 각각의 Inference cycle
    if len(runs) < 2:
        die(f"{v}: runtrace에 Inference가 2개 미만")
    q = read_json(res / f"optrace/{v}/qhas.json")
    try:
        inst = q["data"]["htp_op_instances"]["data"]           # HTP op 목록 (첫 실행)
        overall = q["data"]["htp_overall_summary"]["data"][0]  # 전체 요약 (timeline_cycles, htp_nodes 등)
    except (KeyError, IndexError):
        die(f"{v}: qhas.json 구조가 예상과 다름")
    missing = [k for k in QHAS_FIELDS if k not in inst[0]]
    if missing:
        die(f"{v}: qhas.json op 필드 없음 {missing}")
    return {"trial": trial, "htp": htp, "cycles": st.median(runs[1:]), "inst": inst, "overall": overall}   # Optrace cycle = 첫 회를 뺀 20회 중앙값


def sqnr(res, v):
    # 보드 출력과 FP32 기준값의 SQNR(dB), outputs/가 없으면 "-"
    y, ref = res / f"outputs/{v}.raw", res / "outputs/test_ref_nhwc.raw"
    if not (y.is_file() and ref.is_file()):
        return "-"
    y, ref = np.fromfile(y, np.float32), np.fromfile(ref, np.float32)
    if y.size != 1024 * 256 or ref.size != 1024 * 256:
        die(f"{v}: 출력 크기가 1024x256이 아님")
    y = y.reshape(1024, 256) if v == "v2_nhwc" else y.reshape(256, 1024).T  # v0/v1 출력은 NCHW -> 전치해서 [토큰, 채널]로
    ref = ref.reshape(1024, 256)
    return f"{10 * np.log10((ref ** 2).sum() / ((y - ref) ** 2).sum()):.1f} dB"   # 10 log10(신호 제곱합 / 오차 제곱합)


def draw(data, path):
    # 세 그래프의 Optrace 타임라인(첫 실행)을 그려 저장
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#e4e3df", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "svg.fonttype": "none"})
    font = ROOT / "work/fonts/NanumGothic-Regular.ttf"  # 있으면 한글을 path로 SVG에 넣어 어느 뷰어에서나 같게 보임
    if font.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font))
        plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                             "svg.fonttype": "path"})

    # 세로는 HTP 스레드(256 = HMX·DMA, 512/513 = HVX), 가로는 첫 op 시작을 0으로 한 cycle
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.2), sharex=True)
    for ax, v in zip(axes, VARIANTS):
        inst, overall = data[v]["inst"], data[v]["overall"]
        t0 = min(int(r["start_cycle"]) for r in inst)                   # 가장 먼저 시작한 op를 0으로
        for r in inst:
            lane = {"256": 2, "512": 1, "513": 0}.get(str(r["tid"]))    # 스레드 -> 세로 위치
            if lane is not None:
                ax.broken_barh([((int(r["start_cycle"]) - t0) / 1e3, max(int(r["cycles"]) / 1e3, 0.3))],   # 천 cycle 단위, 짧은 op도 보이게 최소 0.3
                               (lane - 0.36, 0.72), facecolors=CATS[category(r)][1], edgecolor="none")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["HVX 513", "HVX 512", "HMX/DMA 256"], fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e4e3df")
        ax.set_xlim(left=0)
        ax.set_title(f"{v}   {int(overall['timeline_cycles']) / 1e3:.0f}k cycles, HTP op {overall['htp_nodes']}개",
                     loc="left", fontsize=10)                       # 칸 제목: 전체 cycle, HTP op 수 (첫 실행)
    axes[-1].set_xlabel("HTP cycle")
    axes[-1].xaxis.set_major_formatter(lambda x, _: f"{x:.0f}k")    # 눈금을 "50k"처럼
    fig.legend(handles=[Patch(color=c, label=n) for n, c in CATS], loc="upper center", ncol=3, fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None}) # 날짜를 빼서 다시 그려도 내용이 같게


def main():
    # 결과 표 출력, 그림은 기존 파일이 있으면 --overwrite 없이는 그대로 둠 (원본 결과 보존)
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?", default=ROOT / "results/week_1")   # 결과 폴더 (생략하면 results/week_1)
    ap.add_argument("--figure", type=Path)                                              # 그림 경로 (생략하면 결과 폴더 안)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    data = {v: load(a.results, v) for v in VARIANTS}

    print("| 단계 | graphExecute (trial 범위) | HTP 실행 | Optrace cycle | HTP op 수 | SQNR |")
    for v in VARIANTS:
        x = data[v]
        print(f"| {v} | {st.median(x['trial']):.0f} µs ({min(x['trial']):.0f}~{max(x['trial']):.0f}) "   # trial 중앙값들의 중앙값 (최소~최대)
              f"| {st.median(x['htp']):.0f} µs | {x['cycles'] / 1e3:.0f}k | {x['overall']['htp_nodes']} "
              f"| {sqnr(a.results, v)} |")

    fig = a.figure or a.results / "figures/optrace_timeline.svg"
    if fig.exists() and not a.overwrite:
        print(f"[analyze] 그림 유지: {fig} (새로 그리려면 --figure 다른 경로 또는 --overwrite)")
    else:
        draw(data, fig)
        print(f"[analyze] 그림: {fig}")


if __name__ == "__main__":
    main()
