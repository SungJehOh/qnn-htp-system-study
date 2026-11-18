# Week 1: QNN/HTP 실행 구조와 Optrace

## 이번 주 목표

ONNX 모델이 QNN을 거쳐 HTP에서 실행되는 구조를 정리하고, 같은 계산을 하는 그래프 세 개를 Optrace로 비교한다.

- 그래프를 HTP에 맞게 바꾸면(Linear → 1×1 Conv, Transpose 제거) HTP 실행 시간이 1.7배 줄었다.
- 행렬 곱 자체의 비용은 그대로였다. 무엇이 줄었는지를 Optrace와 QHAS에서 찾는다.

측정 환경: Rubik Pi 3 (QCS6490, HTP v68), Qualcomm Linux 1.5, QAIRT 2.35.0 (Host, 보드 모두). 공식 설명은 SDK 문서(`<QAIRT>/docs/QNN/`)로 확인했다.

---

## 1. 배경: 같은 계산, 다른 그래프

ViT, MLP 블록의 FFN 하나를 쓴다.

```text
y = x + W2 · ReLU(W1 · x + b1) + b2      x: 32×32 = 1,024 토큰 × 256 채널, 256 → 1024 → 256
```

PyTorch에서는 보통 이렇게 쓴다.

```python
t = x.flatten(2).transpose(1, 2)     # [1,256,32,32] → [1,1024,256]
y = t + linear2(relu(linear1(t)))
y = y.transpose(1, 2).reshape(1, 256, 32, 32)
```

| 단계 | 그래프 | 입출력 (uint8) | QNN Graph |
|---|---|---|---|
| v0 naive | 위 코드 그대로: Reshape/Transpose → MatMul+Add ×2 → 되돌리기 | NCHW `[1,256,32,32]` | Reshape 6, Transpose 2, FullyConnected 2, ElementWiseNeuron, Eltwise_Binary |
| v1 | Linear를 1×1 Conv로 (4D NCHW 유지) | NCHW | Transpose 2, Conv2d 2, Eltwise_Binary |
| v2 | 1×1 Conv, NHWC 입출력 | NHWC `[1,32,32,256]` | Conv2d 2, Eltwise_Binary |

- 1×1 Conv는 픽셀(토큰)마다 같은 행렬을 채널 벡터에 곱한다. Linear와 같은 계산이고, 가중치 모양만 `[out, in, 1, 1]`로 바꾸면 된다.
- 가중치는 세 그래프가 같다. `make_models.py`가 ONNX Reference로 같은 출력인지 확인한다.
- 양자화는 W8A8, calibration 입력 8개다.
- v0, v1은 `--preserve_io layout`으로 입출력을 NCHW로 유지한다. 그래서 QNN Graph에 입출력 Transpose가 들어간다. v2는 `--input_layout x NHWC`다.

---

## 2. QNN 실행 구조

### Host에서 컴파일, 보드에서 실행

```text
[Host]  ONNX ─ qnn-onnx-converter ──────────────> QNN 그래프 .cpp + 가중치 .bin (W8A8 양자화)
             ─ qnn-model-lib-generator ─────────> 모델 라이브러리 .so (x86)
             ─ qnn-context-binary-generator ────> Context Binary .bin (+ Optrace용 schematic.bin)
[보드]  Context Binary ─ contextCreateFromBinary ─ graphRetrieve ─ graphExecute (반복)
```

- Context Binary = HTP용 컴파일이 끝난 그래프 + 상수. 변환기의 가중치 `.bin`과는 다른 파일이다.
- 보드에서 컴파일하는 방법(online prepare)도 있지만 이 실습은 쓰지 않는다.

### QNN 객체와 이 실습의 파일

| 구성 요소 | 의미 (API overview 요약) | 이 실습의 파일, 코드 |
|---|---|---|
| Backend | 최상위 객체. 대부분의 API는 Backend를 먼저 만들어야 쓸 수 있다 | 보드 `/usr/lib/libQnnHtp.so`를 `bench_graph`가 `dlopen` → `backendCreate` |
| Device | 실행에 쓸 하드웨어 자원을 가리키는 handle. 전력 제어도 여기서 한다 | 파일 없음. `deviceCreate` 후 `setPowerConfig`로 burst 설정 (`qnn-net-run`은 `htp.json`) |
| Context | 그래프의 실행 환경. binary로 저장했다가 다시 로드할 수 있다 | `ctx/v0_naive.bin` (Host `work/qnn/v0_naive/plain/`에서 생성), Optrace용 `ctx/v0_naive_optrace.bin` |
| Graph | Context 안의 노드와 텐서 묶음 | Context 파일 안에 들어 있다. 이름 `v0_naive` (`make_models.py`가 붙인 이름)로 `graphRetrieve` |
| Tensor | 상수 또는 입출력 데이터 | 입력 `x` ← `inputs/test_nchw.raw`(float32 → uint8), 출력 `y` → `out/v0_naive.raw`. shape, scale/offset 정의는 Context 안 |

### Backend 라이브러리와 HTP Runtime은 다른 프로세서에서 돈다

- QCS6490 안에는 Linux가 도는 Arm CPU와, 별도 프로세서인 Hexagon cDSP(HTP)가 있다.
- 보드에서 cDSP는 Linux의 remoteproc가 펌웨어 `qcom/qcs6490/cdsp.mdt`를 올려 따로 실행한다.

```text
Arm CPU (Linux, aarch64)                                  Hexagon cDSP = HTP (DSP6, 32-bit)
bench_graph 프로세스 (taskset -c 7 → cpu7)
 ├ libQnnHtp.so          QNN API 구현 (backendCreate ...)
 ├ libQnnSystem.so       Context Binary 정보 읽기
 └ libQnnHtpV68Stub.so ── FastRPC (/dev/fastrpc-cdsp) ──> libQnnHtpV68Skel.so, libQnnHtpV68.so
   위치: /usr/lib (LD_LIBRARY_PATH)                         위치: /usr/lib/rfsa/adsp (ADSP_LIBRARY_PATH)
                                                           그래프 실행: HMX, HVX, DMA, VTCM
```

- `file`로 보면 `libQnnHtp.so`, `Stub`은 `ARM aarch64`, `Skel`은 `QUALCOMM DSP6` 바이너리다. Skel은 CPU에서 실행할 수 없다.
- 그래서 보드에서 경로를 두 개 지정한다. `LD_LIBRARY_PATH`는 CPU 쪽, `ADSP_LIBRARY_PATH`는 cDSP 쪽이다.
- `taskset -c 7`은 CPU 쪽 프로세스만 고정한다. 그래프 실행은 cDSP에서 일어난다.
- `graphExecute`는 CPU가 FastRPC로 cDSP에 실행을 요청하고 기다리는 호출이다. 그래서 1회 시간에 CPU, FastRPC, HTP 시간이 모두 들어간다(4장).

### `bench_graph.cpp`의 호출 순서

"(추정)"은 공개 자료로 확인하지 못한 내부 동작이다.

| # | 호출 | 하는 일 | 생기는 것 |
|---|---|---|---|
| 1 | `dlopen`, `QnnInterface_getProviders` | 라이브러리를 열고 API 함수 테이블을 얻는다 | 함수 포인터 `q.*` |
| 2 | `systemContextGetBinaryInfo` | Context Binary에서 그래프 정보를 읽는다 (2.35에서 deprecated) | 그래프 이름, 입출력 텐서 정의 |
| 3 | `backendCreate` | Backend 초기화 | Backend handle |
| 4 | `deviceCreate`, `setPowerConfig` | Device handle 생성, HTP 전력 burst | Device handle |
| 5 | `contextCreateFromBinary` | binary로 Context 생성. HTP 쪽에 그래프를 올린다(추정, profile에 `Accelerator (load binary) time` 이벤트가 있음) | Context handle |
| 6 | `graphRetrieve` | 이름으로 그래프를 찾는다 | Graph handle |
| 7 | 텐서에 버퍼 연결 | 텐서 정의에 앱이 할당한 RAW 버퍼를 붙인다. 실행마다 HTP가 읽을 수 있는 메모리로 옮겨진다(추정) | 입출력 Tensor |
| 8 | `graphExecute` | 그래프를 동기 실행한다 | 출력 버퍼에 결과 |
| 9 | `contextFree`, `deviceFree`, `backendFree` | 해제 | — |

- QNN 실행 구조는 API 객체의 준비 순서다.
- HTP 하드웨어 실행 구조(3장)는 `graphExecute` 한 번 안에서 cDSP가 하는 일이다.
- QNN Graph 하나가 HTP 안에서는 63~226개의 HTP op가 된다.

---

## 3. HTP 구조

### 자원

```text
System DRAM ─ DMA ─ VTCM (칩 안 메모리, 이 보드 2 MB) ─┬─ HMX (행렬)
                                                      └─ HVX (벡터)
```

| 자원 | 역할 | 이 실습에서 |
|---|---|---|
| HMX | 행렬 곱 | Conv2d, FullyConnected 모두 `q::ConvLayer_s1.opt` (스레드 256) |
| HVX | 벡터 연산 | Transpose, 포맷 변환, Add (스레드 512, 513) |
| VTCM | 연산 직전 데이터를 두는 메모리 | 입력, 가중치, 중간 결과 |
| DMA | DRAM ↔ VTCM 복사 | `*InputSlice`, `weights_to_vtcm`, `*OutputSlice` |

VTCM 2 MB는 QNN API `deviceGetPlatformInfo`로 확인했다.

### 연산 하나가 실행되는 흐름

1. 입력은 System DRAM에 있다.
2. DMA가 입력 조각과 가중치를 VTCM으로 옮기고, 다음 op는 `DmaCheckpointWait`로 기다린다.
3. HVX가 데이터를 HTP 내부 배치로 바꾸고(`ForceFormat_Crouton`), HMX가 행렬 곱, HVX가 Add를 한다.
4. 결과는 다음 op의 입력이 되거나 DMA가 DRAM 출력 버퍼로 쓴다.

- v2 Optrace의 실제 순서: `InputSlice` → `ForceFormat_Crouton`/`SlicePad` → `ConvLayer` → `addsub_op` → `OutputSlice`
- 개념적인 흐름이다. 실제로는 텐서를 조각내 단계들을 겹쳐 실행하고, 조각 수와 순서는 컴파일 때 정해진다.

### 헷갈리기 쉬운 것

| 구분 | 차이 |
|---|---|
| HMX / HVX | HMX는 행렬 곱 전용. 배치 변환처럼 행렬 곱이 아닌 일은 HVX가 한다 |
| VTCM / DRAM | HMX, HVX는 VTCM을 읽는다. 중간 텐서가 VTCM에 안 들어가면 DRAM으로 내보냈다 다시 읽는다(`TcmToDram`/`DramToTcm`). 이번 세 그래프에는 없다 |
| DMA 전송 / 동기화 | 전송은 데이터 이동. `DmaCheckpointSet/Wait`, `SyncOp`는 순서 맞추기 (공식 문서의 System Service). Wait가 길면 데이터가 늦게 온 것 |
| Layout 변환 | 그래프 입출력은 flat(NCHW/NHWC), HTP 안은 내부 타일 배치(`ForceFormat_*`). 그래프의 Transpose는 `Transpose_impl`, `Transpose.2D`가 된다. 타일 크기는 공개 자료로 확인하지 못했다 |

---

## 4. Profiling과 Optrace

### 도구 비교

| 도구 | 단위 | 보는 것 | 못 보는 것 |
|---|---|---|---|
| Host Timer | `graphExecute` 1회 | 앱이 체감하는 시간 | 시간이 어디서 쓰였는지 |
| `perf` | CPU 함수 | QNN, FastRPC의 CPU 쪽 비용 | HTP 안 (이 실습에서는 안 씀) |
| QNN Basic | Graph | Host, RPC, HTP 구간별 시간 | op별 내용 |
| QNN Detailed | QNN Op | op별 cycle | 시간 순서, 겹침. cycle을 µs로 바로 바꿀 수 없다(공식 문서) |
| QNN Linting | QNN Op | main thread 기준 실행, Wait, Overlap cycle, 자원 | HTP op 단위 타임라인 (이 실습에서는 안 씀) |
| HTP Optrace | HTP Op | op별 시작 cycle, 길이, 스레드, 자원, 메모리 바이트, runtrace | CPU, FastRPC, 명령어 단위, SoC 전체 |
| QHAS | HTP Op 요약 | Optrace를 HTP op, QNN op로 합친 표, dominant path (Beta) | Optrace와 같음 |

Optrace는 명령어 단위 tracer도, SoC 전체 profiler도 아니다. 컴파일된 HTP op가 cDSP 안에서 언제 어떻게 실행됐는지를 본다.

### `graphExecute` 시간의 구성

공식 문서 "HTP Execute Basic Profiling Events" 그림 기준이다.

| Basic profile 이벤트 | 범위 |
|---|---|
| `QNN (execute) time` | `graphExecute` 전체 |
| `RPC (execute) time` | FastRPC 요청 ~ 응답 |
| `QNN accelerator (execute) time` | HTP 쪽 실행 준비 ~ 반환 |
| `Accelerator (execute) time` | HTP 자원 확보 + 그래프 실행 + 자원 반환. 이 자료의 "HTP 실행" |

### Optrace 수집

```text
[Host]  qnn-context-binary-generator ... --profiling_level detailed --profiling_option optrace
        → Context Binary(계측 포함) + schematic.bin
[보드]  qnn-net-run ... --profiling_level detailed --profiling_option optrace --num_inferences 21
        → qnn-profiling-data_0.log
[Host]  qnn-profile-viewer --reader libQnnHtpOptraceProfilingReader.so --schematic ... --config reader.json
        → optrace.json (Perfetto나 QAIRT Visualizer에서 열기), QHAS(qhas.json), runtrace(runtrace.json)
```

- schematic은 같은 Context를 만들 때 생긴 것이어야 한다. `reader.json`에서 `qhas_json`, `runtrace`를 켠다.
- 보드는 21회 모두 op 단위로 기록하지만, qnn-profile-viewer는 op 단위 상세(Chrome trace, QHAS)를 첫 실행만 풀어 준다. runtrace만 21회 모두 나온다.
- Optrace를 켠 실행은 trace 전송 때문에 `graphExecute`가 수 ms가 된다. 시간은 Optrace 없는 Context로 따로 잰다.

### ONNX Op ≠ QNN Op ≠ HTP Op

변환기와 HTP 컴파일러가 합치고(Fusion), 나누고(Splitting), 배치 변환을 넣거나 없앤다. v0의 실제 대응:

| ONNX | QNN Graph (`*_net.json`) | QHAS의 HTP op |
|---|---|---|
| Reshape, Transpose | `Reshape_0`, `Transpose_0` | `Transpose_0`: `Transpose_impl` 32, `SlicePad` 32, `ForceFormat_Crouton` 32 |
| MatMul, Add | `MatMul_0`(FullyConnected) + Reshape 2 | 없음 |
| Relu | `Relu_0` | `Relu_0`: `ConvLayer` 4 + 가중치 DMA |
| MatMul, Add | `MatMul_1`(FullyConnected) + Reshape 2 | `MatMul_1`: `ConvLayer` 8 + 가중치 DMA |
| Add | `Add_2` | `Add.tcm` 8 등 |
| Transpose, Reshape | `Transpose_1`, `Reshape_1` | `Transpose_1`: `Transpose_impl` 8, `SlicePad` 8, `Concat` 1 |

- QNN 노드 12개 중 HTP op가 붙은 것은 5개다. Reshape는 HTP op로 남지 않았다.
- 첫 행렬 곱이 `MatMul_0`이 아니라 `Relu_0` 이름으로 표시된다. FullyConnected와 ReLU를 합친 것으로 해석한다.
- QHAS의 `qnn_op`는 컴파일러가 기록한 소속이라 소스 그래프 경계와 다를 수 있다. 예: v1의 출력 Transpose 작업은 `Output`으로 표시된다.

---

## 5. QHAS 읽기

### 주요 필드

`qhas.json`의 `data.htp_op_instances.data`가 HTP op 목록이다(첫 실행).

| 필드 | 의미 |
|---|---|
| `htp_op`, `qnn_op` | HTP op 이름, 소속 QNN op |
| `start_cycle`, `cycles` | 시작 시각, 실행 길이 (cycle) |
| `num_dominant_path_cycles` | dominant path에 있었던 cycle |
| `tid` | HTP 스레드 (256: HMX·DMA, 512/513: HVX) |
| `hmx`, `hvx`, `dma` | 쓴 자원 (`dma_wait`, `sync` 등도 있다) |
| `dram_read/write`, `vtcm_read/write` | 읽고 쓴 바이트 |

같은 op끼리 합친 표는 `htp_op_types`, QNN op별 합계는 `qnn_op_instances_nodes`, 전체 cycle과 op 수는 `htp_overall_summary`(`timeline_cycles`, `htp_nodes`)에 있다.

### dominant path

공식 문서: 매 순간 실행 중인 op 가운데 우선순위가 가장 높은 op에 그 cycle을 준다.

```text
HMX > HVX > DMA 읽기 > DMA 쓰기 > DmaCheckpointSet/Wait > SyncOp
```

- 그래서 `num_dominant_path_cycles` 합은 전체 길이와 거의 같다 (v0: 406,918 / 409,641). `cycles` 합은 겹친 시간이 중복돼 610,606이다.
- 의존 관계로 구한 critical path가 아니다. 같은 우선순위끼리 겹칠 때 cycle을 누가 갖는지는 문서로 확인하지 못했다.

### 예제: v0의 `q::Transpose_impl`

| 항목 | 값 (JSON에서 확인) |
|---|---|
| 개수, 소속 | 40개 (`Transpose_0` 32, `Transpose_1` 8) |
| 스레드, 자원 | 512에 20개, 513에 20개. 모두 HVX만 사용 |
| 메모리 | DRAM 0, VTCM 읽기·쓰기 각 524,288 B |
| `cycles` 합 / dominant 합 | 216,717 / 109,325 (dominant path의 26.9%) |

1. 출처: ONNX Transpose 두 개(`transpose(1, 2)`와 되돌리기)가 HTP에서 40조각이 됐다.
2. 자원: HVX만 쓰고 DRAM 접근이 없다. VTCM 안에서 배치를 바꾸는 작업이다.
3. cycle 차이:
   - 512와 513이 하나씩 동시에 실행해 20개만 dominant path에 잡혔다(나머지 20개는 거의 0).
   - 실행 시간 216,717 cycle 중 207,037 cycle이 다른 HVX 스레드와 겹친다.
4. 겹침:
   - HMX 스레드와 겹친 시간은 4,347 cycle뿐이다.
   - `Transpose_0`은 43k~169k cycle에 실행되고, 첫 `ConvLayer`는 177k에 시작한다.
5. 판단:
   - 확인된 것: Transpose가 도는 동안 HMX는 쉬었고, Transpose가 dominant path의 27%다.
   - 가설: 이것을 없애면 HMX가 일찍 시작해 전체가 짧아진다. 7장(v1 결과)에서 확인한다.
   - 주의: `cycles`가 큰 op가 곧 병목은 아니다. v2 `ForceFormat_Crouton`은 `cycles` 53,962인데 dominant는 6,370이다.

---

## 6. 측정 방법

| 측정 | 방법 | 워밍업 | 표의 값 |
|---|---|---|---|
| graphExecute | `bench_graph`, 300회 × 프로세스 10개 | 프로세스마다 100회 버림 | trial 중앙값들의 중앙값 (괄호: 최소~최대) |
| HTP 실행 | 같은 프로세스, basic profile 20회 | 위 400회 뒤 | `Accelerator (execute) time` 중앙값 |
| Optrace cycle | `qnn-net-run` 21회 | 첫 1회 버림 | runtrace 20회 중앙값 |
| 타임라인 그림 | 같은 실행의 QHAS | 없음 | 첫 실행 |

- 공통: HTP 전력 burst, `bench_graph`는 `taskset -c 7`로 큰 CPU 코어에 고정 (CPU 쪽 시간이 코어에 따라 200 µs 이상 달랐다).
- 첫 실행 효과: runtrace 첫 회는 v0 413k, v1 294k로 이후보다 약 10% 많다(v2는 5% 적다). 이후 20회는 ±2% 안이다.
- 표의 cycle(워밍업 후)과 타임라인 그림(첫 실행)은 다른 실행의 값이다.
- 변동: HTP 실행은 모든 값이 중앙값의 -2~+4% 안이다. graphExecute는 trial마다 크게 흔들렸고(v0 558~899 µs) 원인은 확인하지 못했다.

---

## 7. 결과와 해석

### 결과

`python3 tools/analyze.py results/week_1`로 원본에서 다시 계산한 값이다.

| 단계 | graphExecute (trial 범위) | HTP 실행 | Optrace cycle | HTP op 수 |
|---|---:|---:|---:|---:|
| v0 naive | 650 µs (558~899) | 269 µs | 376k | 226 |
| v1 1×1 Conv | 542 µs (490~571) | 197 µs | 267k | 63 |
| v2 NHWC | 470 µs (449~542) | 160 µs | 218k | 121 |

![Optrace 타임라인](../../../results/week_1/figures/optrace_timeline.svg)

정확도 (측정 당시 보드 출력, 파일은 저장소에 없음):

- 세 그래프 모두 SQNR 34.2 dB, 최대 오차 0.164다.
- v1과 v2는 비트 단위로 같다.
- v0은 262,144개 중 6개 값이 양자화 1단계 다르다(출력 scale이 소수점 8번째 자리에서 다름).

### 무엇이 바뀌었나

QHAS(첫 실행)로 HMX 구간과 그 앞뒤를 나눈 값이다.

| 단계 | `ConvLayer` dominant | 첫 `ConvLayer` 시작 | 마지막 `ConvLayer` 이후 | HVX `cycles` 합 | 배치 변환 dominant |
|---|---:|---:|---:|---:|---:|
| v0 | 133,505 | 177k | 94k | 419k | 218k |
| v1 | 132,733 | 89k | 68k | 158k | 106k |
| v2 | 132,299 | 48k | 24k | 140k | 37k |

- 행렬 곱 비용은 같다(0.9% 차이).
- 줄어든 것은 HMX 앞뒤를 채우던 HVX 배치 변환이다(`Transpose*`, `ForceFormat*`, `SlicePad*`, `Concat`, `Reshape*`).

| 변경 | 사라진 것 | 남은 것, 주의할 점 |
|---|---|---|
| v0 → v1 | 그래프 안 Transpose의 `Transpose_impl` 40, `SlicePad` 40, `Concat`. HTP op 226 → 63 | 입출력 `Transpose.2D` 4개(dominant 26.9%). 연산 표현(FC → Conv)과 배치 처리(3D 토큰 ↔ 4D)가 함께 바뀌어 원인을 하나로 나눌 수 없다 |
| v1 → v2 | 입출력 `Transpose.2D`. `ConvLayer`가 dominant의 64.6% | HTP op는 63 → 121로 늘었다(`InputSlice` 1→4, `OutputSlice` 1→8, `ForceFormat_*` 1→12, `SlicePad` 0→8). 왜 이렇게 나눴는지는 확인하지 못했다. op 수로 성능을 판단하면 안 된다 |

v1만 마지막 `*OutputSlice`가 HVX(스레드 512)다. v0, v2는 DMA(스레드 256)다. 직전 Transpose를 HVX가 하기 때문으로 보이지만 확인하지 못했다.

### graphExecute와 HTP 실행의 차이

| 단계 | graphExecute − HTP 실행 |
|---|---:|
| v0 | 381 µs |
| v1 | 345 µs |
| v2 | 310 µs |

- 그래프를 바꿔도 이 차이는 거의 줄지 않는다. 그래서 graphExecute 개선(1.4배)이 HTP 실행 개선(1.7배)보다 작다.
- 들어갈 수 있는 것(4장 basic profile 그림 기준): Host 쪽 QNN 처리, RAW 버퍼 처리, FastRPC 요청·응답, HTP 쪽 실행 준비·반환
- `results/week_1`에는 graphExecute와 `Accelerator (execute) time`만 있어서 항목별 비중은 모른다.
- 현재 `bench_graph`는 `QNN (execute)`, `RPC (execute)`, `QNN accelerator (execute)`도 기록한다. 다음 측정부터 나눠 볼 수 있다.

---

## 8. 정리

- QNN은 Host 컴파일과 보드 실행으로 나뉜다. 보드는 Context Binary를 로드해 `graphExecute`만 부른다.
- QNN API는 CPU에서, 그래프는 cDSP에서 돈다. 둘은 FastRPC로 이어지고, `graphExecute` 시간에는 양쪽이 다 들어간다.
- HTP에서 Linear도 HMX로 실행된다. PyTorch식 그래프가 느렸던 것은 Linear 때문이 아니라 그 앞뒤의 Transpose가 HVX 배치 변환이 되어 HMX 시작을 늦췄기 때문이다.
- Optrace는 cycle이 큰 op가 아니라 dominant path로 읽는다. 겹쳐 숨은 시간은 병목이 아니다. op 수가 늘어도 빨라질 수 있다.
- 확인하지 못한 것:
  - graphExecute − HTP 실행 차이의 항목별 비중
  - graphExecute가 프로세스마다 흔들리는 원인
  - 크기를 바꿔도 같은 경향인지
  - dominant path에서 같은 우선순위끼리 cycle을 나누는 규칙
  - cycle → µs 환산

## 참고 자료

- `<QAIRT>/docs/QNN/general/htp/htp_backend.html`: HTP Backend, Basic/Detailed/Linting Profiling, Optrace, QHAS ([온라인](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html))
- `<QAIRT>/docs/QNN/general/api_overview.html`: Backend, Device, Context, Graph, Tensor 정의
- `<QAIRT>/include/QNN/` 헤더 (`QnnGraph.h`, `QnnContext.h`, `HTP/QnnHtpPerfInfrastructure.h`): API 설명, 전력 설정
- [qai-appbuilder Profiling Reference](https://github.com/qualcomm/qai-appbuilder/blob/main/tools/skills/qai-runner-skill/references/profiling.md): `qnn-profile-viewer`, Optrace Reader, QHAS 사용 예
- [toy-infer-runtime](https://github.com/moons98/toy-infer-runtime): 주차 구성 참고
