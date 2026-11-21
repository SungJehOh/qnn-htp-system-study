#!/usr/bin/env python3

"""같은 FFN 블록을 세 가지 ONNX 그래프로 만든다.

    y = x + W2 · ReLU(W1 · x + b1) + b2      x: 32x32 = 1024 토큰 x 256 채널, 256 -> 1024 -> 256

v0_naive   : NCHW -> flatten + Transpose -> MatMul(Linear) x2 -> 되돌리기   (PyTorch에서 흔히 쓰는 모양)
v1_conv1x1 : NCHW 4D 그대로 Linear를 1x1 Conv로
v2_nhwc    : 1x1 Conv, 그래프 입출력이 NHWC
"""
# 파일 전체 역할
# 1. 같은 가중치(w1, b1, w2, b2)로 FFN 블록을 세 가지 ONNX 그래프로 생성 (계산은 같고 모양만 다름)
# 2. QNN 변환기의 양자화(calibration)에 쓸 입력 8개와 측정용 입력 1개를 raw 파일로 저장
# 3. 측정용 입력의 FP32 기준 출력(numpy로 직접 계산)을 저장 -> 보드 출력의 정확도 비교용
# 4. ONNX Reference 실행기로 세 그래프가 기준 출력과 같은 값을 내는지 확인
# 실행: python3 tools/make_models.py <출력 폴더>   (run.sh가 work/models로 부른다)

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

# ===== 크기 설정 =====
H = W = 32          # 이미지 높이, 너비 (32 x 32)
C, HID = 256, 1024  # 채널 수 256, FFN 중간(hidden) 차원 1024
N = H * W           # 토큰 수 = 픽셀 수 = 1024

out = Path(sys.argv[1])                              # 첫 번째 명령행 인자를 출력 폴더 경로로 사용
(out / "inputs").mkdir(parents=True, exist_ok=True)  # <출력 폴더>/inputs 생성, parents=True -> 상위 폴더도 같이 생성


# 시드를 고정한 난수 생성기
rng = np.random.default_rng(2026)

# 가중치 생성
w1 = (rng.normal(0, 1, (C, HID)) / np.sqrt(C)).astype(np.float32)    # 첫 번째 Linear 가중치 [256, 1024]
b1 = rng.normal(0, 0.1, HID).astype(np.float32)                      # 첫 번째 Linear bias [1024]
w2 = (rng.normal(0, 1, (HID, C)) / np.sqrt(HID)).astype(np.float32)  # 두 번째 Linear 가중치 [1024, 256]
b2 = rng.normal(0, 0.1, C).astype(np.float32)                        # 두 번째 Linear bias [256]

# numpy 배열 a를 이름이 name인 ONNX 상수 텐서(initializer)로 변환
# np.ascontiguousarray -> 전치(.T) 등으로 메모리 배치가 뒤섞인 배열을 연속된 배열로 복사
T = lambda name, a: numpy_helper.from_array(np.ascontiguousarray(a), name)
node = helper.make_node  # node("Name", [입력], [출력])로 ONNX 노드 생성


def save(name, nodes, in_shape, inits):
    # 노드 목록으로 ONNX 그래프와 모델을 만들고 <name>.onnx로 저장
    # 입력 x와 출력 y는 같은 shape이다.
    g = helper.make_graph(
        nodes,                                                                  # 그래프를 이루는 노드 목록 (실행 순)
        name,                                                                   # ONNX 그래프 이름 (QNN 그래프 이름은 run.sh 변환기의 --output_path 파일 이름으로 정해짐)
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],      # 그래프 입력 x
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, in_shape)],      # 그래프 출력 y
        inits)                                                                  # 상수 텐서 목록 (가중치, bias, reshape용 shape)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])       # 그래프를 모델로 감쌈
    m.ir_version = 7
    onnx.checker.check_model(m)                                                 # 모델 구조가 ONNX 규칙에 맞는지 검사
    onnx.save(m, out / f"{name}.onnx")                                          # <출력 폴더>/<name>.onnx로 저장


# v0: PyTorch식 그래프
# x.flatten(2).transpose(1, 2) -> Linear -> ReLU -> Linear -> +x (residual) -> 이미지로 되돌리기
save("v0_naive", [
    node("Reshape", ["x", "s_tok"], ["xf"]),               # [1,256,32,32] -> [1,256,1024]: flatten
    node("Transpose", ["xf"], ["t"], perm=[0, 2, 1]),      # [1,256,1024] -> [1,1024,256]: 축 순서 바꿈, 토큰마다 256채널 벡터가 되게
    node("MatMul", ["t", "w1"], ["h0"]),                   # [1,1024,256] x [256,1024] -> [1,1024,1024]: 첫 번째 Linear의 행렬 곱
    node("Add", ["h0", "b1"], ["h"]),                      # bias 더하기
    node("Relu", ["h"], ["r"]),                            # ReLU
    node("MatMul", ["r", "w2"], ["f0"]),                   # [1,1024,1024] x [1024,256] -> [1,1024,256]: 두 번째 Linear
    node("Add", ["f0", "b2"], ["f"]),                      # 두 번째 bias
    node("Add", ["f", "t"], ["o"]),                        # residual: FFN 결과 + 입력 토큰
    node("Transpose", ["o"], ["of"], perm=[0, 2, 1]),      # [1,1024,256] -> [1,256,1024]: 축 순서 되돌림
    node("Reshape", ["of", "s_img"], ["y"]),               # [1,256,1024] -> [1,256,32,32]: 이미지 모양
], [1, C, H, W],                                           # 그래프 입출력 shape: [N, C, H, W] = [1,256,32,32]
   [T("w1", w1), T("b1", b1), T("w2", w2), T("b2", b2),    # 가중치와 bias
    T("s_tok", np.array([1, C, N], np.int64)),             # Reshape가 쓸 목표 shape [1,256,1024] (ONNX Reshape는 shape를 int64 텐서로 받음)
    T("s_img", np.array([1, C, H, W], np.int64))])         # 되돌릴 때 쓸 목표 shape [1,256,32,32]



# v1, v2: Linear를 1x1 Conv로
# 1x1 Conv는 토큰마다 같은 행렬을 채널 벡터에 곱하는 연산 -> Linear와 같은 계산
# Linear 가중치 w1 [in=256, out=1024]를 Conv 커널 모양 [out, in, 1, 1]로 바꿈
# w1.T -> 전치해서 [1024, 256], reshape(HID, C, 1, 1) -> [1024, 256, 1, 1]
conv = [T("w1", w1.T.reshape(HID, C, 1, 1)),   # 첫 번째 1x1 Conv 커널 [1024, 256, 1, 1]
        T("b1", b1),                           # Conv bias는 출력 채널마다 하나 [1024]
        T("w2", w2.T.reshape(C, HID, 1, 1)),   # 두 번째 1x1 Conv 커널 [256, 1024, 1, 1]
        T("b2", b2)]                           # [256]

# FFN 본체를 만드는 함수: 입력 텐서 이름 i, 출력 텐서 이름 o를 받아 노드 4개 목록을 돌려줌
# v1과 v2가 같은 본체를 쓰고 앞뒤만 다르므로 함수로 묶음
ffn = lambda i, o: [node("Conv", [i, "w1", "b1"], ["h"]),       # 1x1 Conv: 256 -> 1024 채널 (bias 포함)
                    node("Relu", ["h"], ["r"]),                 # ReLU
                    node("Conv", ["r", "w2", "b2"], ["f"]),     # 1x1 Conv: 1024 -> 256 채널
                    node("Add", ["f", i], [o])]                 # residual: 결과 + 입력

# v1: 입출력은 v0과 같은 NCHW, 그래프 안에 flatten/Transpose가 없음
save("v1_conv1x1", ffn("x", "y"), [1, C, H, W], conv)

# v2: 입출력이 NHWC [1,32,32,256]
# ONNX Conv는 NCHW만 받으므로 앞뒤를 Transpose로 감싼다. QNN 변환기에 --input_layout x NHWC를 주면 이 Transpose가 사라진다.
save("v2_nhwc",
     [node("Transpose", ["x"], ["xc"], perm=[0, 3, 1, 2])]      # NHWC -> NCHW
     + ffn("xc", "yc")                                          # 리스트 + 리스트 -> 이어 붙인 리스트
     + [node("Transpose", ["yc"], ["y"], perm=[0, 2, 3, 1])],   # NCHW -> NHWC
     [1, H, W, C], conv)                                        # 그래프 입출력 shape: NHWC



# 입력 데이터
# calibration 8개 + 측정용 test 1개
# 같은 데이터를 NHWC(v2 입력)와 NCHW(v0, v1 입력) 두 배치로 저장
for name in [f"calib{k}" for k in range(8)] + ["test"]:
    # N(0, 1)에서 (N, C) shape의 난수 생성 -> float32로 변환
    x = rng.normal(0, 1, (N, C)).astype(np.float32)      # [1024 토큰, 256 채널] -> 메모리 배치가 그대로 NHWC [1,32,32,256]

    x.tofile(out / f"inputs/{name}_nhwc.raw")            # .tofile -> 헤더 없이 값만 이어서 저장 (raw 파일)

    # v0, v1 입력용 NCHW 배치
    x.T.copy().tofile(out / f"inputs/{name}_nchw.raw")   # 전치 [256, 1024] -> NCHW [1,256,32,32] 배치, .copy()로 연속 배열을 만든 뒤 저장

ref = x + np.maximum(x @ w1 + b1, 0) @ w2 + b2                    # x는 반복의 마지막 = test 입력
ref.astype(np.float32).tofile(out / "inputs/test_ref_nhwc.raw")   # 기준 출력은 NHWC 배치로 저장

for lay in ("nhwc", "nchw"):
    (out / f"calib_{lay}.txt").write_text("".join(f"{(out / f'inputs/calib{k}_{lay}.raw').resolve()}\n" for k in range(8)))



# 세 모델이 같은 함수인지 확인
for name, inp, back in [("v0_naive", x.T.reshape(1, C, H, W), lambda y: y.reshape(C, N).T),     # NCHW 입력, NCHW 출력 -> 전치해서 토큰 배치로
                        ("v1_conv1x1", x.T.reshape(1, C, H, W), lambda y: y.reshape(C, N).T),
                        ("v2_nhwc", x.reshape(1, H, W, C), lambda y: y.reshape(N, C))]:         # NHWC는 메모리 배치가 이미 토큰 배치
    
    # ReferenceEvaluator(모델 경로).run(출력 이름 목록(None = 전부), {입력 이름: 값}) -> 출력 리스트, [0]은 첫 번째 출력 y
    y = ReferenceEvaluator(str(out / f"{name}.onnx")).run(None, {"x": inp})[0]

    # 기준 값과 차이 비교
    assert np.abs(back(y) - ref).max() < 1e-3, name
    
print(f"[OK] {out}: v0_naive / v1_conv1x1 / v2_nhwc (같은 함수 확인)")
