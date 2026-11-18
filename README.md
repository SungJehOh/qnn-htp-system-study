# QNN/HTP 시스템 스터디

Qualcomm QNN으로 모델을 HTP(Hexagon NPU)에서 실행하고, HTP Optrace로 HTP 안에서 무슨 일이 일어나는지 보는 스터디입니다.

- 1주차: QNN·HTP 구조와 Optrace. 같은 FFN 블록을 Linear → 1×1 Conv, Transpose 제거로 바꾸며 Optrace 변화를 본다.
  - [guide.md](docs/weeks/week-1/guide.md): 각자 공부할 때 보는 안내 (개념, 실행, 측정해볼 것, 기준 측정값)
  - [talk.md](docs/weeks/week-1/talk.md): 발표 자료 (구조, Optrace 읽는 법, 결과 해석)

## 구조

```text
docs/weeks/week-1/guide.md   1주차 공부 안내
docs/weeks/week-1/talk.md    1주차 발표 자료
tools/run.sh                 [Host] 모델 생성 → HTP Context → 보드 측정 → Optrace 디코딩 → 결과
tools/make_models.py         ONNX 세 가지 생성
tools/board.sh               [보드] 시간 측정, Optrace 실행
tools/bench_graph.cpp        [보드] graphExecute 시간 측정 러너
tools/analyze.py             결과 표, 타임라인 그래프
results/week_1/              1주차 측정 원본: latency CSV, Optrace(optrace.json은 Perfetto나 QAIRT Visualizer에서 열기), 그래프
```

## 준비물

Host (모델 변환, Context 생성, Optrace 디코딩)

- Linux x86_64, Ubuntu 22.04와 Python 3.10. QAIRT의 x86 도구가 이 버전을 기준으로 만들어져 있습니다. 다른 OS라면 Ubuntu 22.04 컨테이너를 씁니다.
- QAIRT(Qualcomm AI Runtime) SDK 2.35: [Qualcomm Software Center](https://softwarecenter.qualcomm.com/)에서 받아 압축을 풀고 의존 패키지를 설치합니다.
  ```bash
  source <QAIRT>/bin/envsetup.sh                    # QNN_SDK_ROOT, PATH, PYTHONPATH 설정
  sudo bash $QNN_SDK_ROOT/bin/check-linux-dependency.sh
  python3 $QNN_SDK_ROOT/bin/check-python-dependency  # 변환기가 쓰는 Python 패키지
  pip install onnx==1.16.1 matplotlib
  ```
- aarch64 교차 컴파일러: `sudo apt install g++-aarch64-linux-gnu`. 보드용 SDK(eSDK)가 있으면 그것을 써도 됩니다.
- 보드에 비밀번호 없이 접속되는 `ssh`/`scp`(키 등록)

보드 (실행, 측정)

- HTP가 있는 Qualcomm 보드와 Linux. 이 저장소의 결과는 Rubik Pi 3(QCS6490, HTP v68)입니다.
- QNN Runtime: Qualcomm Linux에는 OS에 들어 있습니다(`/usr/lib`, `/usr/lib/rfsa/adsp`). 없으면 `COPY_QNN_RUNTIME=1`로 실행하면 QAIRT의 보드용 Runtime을 같이 보냅니다.
- FastRPC(`libcdsprpc.so`, `/dev/fastrpc-cdsp`), `taskset`, root 권한

## 실행

```bash
source <QAIRT>/bin/envsetup.sh
BOARD=root@<보드 IP> bash tools/run.sh
```

약 1분 걸립니다. 결과는 새 폴더 `work/runs/<UTC 시각>/`에 생기고, 결과 표가 화면에 출력됩니다.
`work/`는 Git이 추적하지 않습니다. `results/week_1/`은 1주차 자료에 쓴 측정 원본이라 스크립트가 바꾸지 않습니다.

저장된 결과는 보드 없이 다시 분석할 수 있습니다. 입력 파일은 읽기만 하고, 그림은 이미 있으면 다시 쓰지 않습니다.

```bash
python3 tools/analyze.py results/week_1                       # 결과 표 출력
python3 tools/analyze.py results/week_1 --figure /tmp/t.svg   # 타임라인 그림을 다른 경로에
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `BOARD` | `rubikpi` | 보드 ssh 대상 |
| `BOARD_DIR` | `/home/root/qnn-htp-system-study` | 보드 작업 폴더 (실행할 때마다 지우고 새로 만듦) |
| `SOC_ID`, `DSP_ARCH` | `35`, `v68` | QNN HTP 대상 SoC. QCS6490 기준. Linux의 `/sys/devices/soc0/soc_id`와 다른 번호 |
| `ESDK_ROOT` | (없음) | 지정하면 보드 eSDK로, 없으면 `AARCH64_CXX`(기본 `aarch64-linux-gnu-g++`)로 러너를 빌드 |
| `COPY_QNN_RUNTIME` | `0` | `1`이면 QAIRT의 보드용 Runtime(`aarch64-oe-linux-gcc11.2`, `hexagon-<arch>`)을 보드로 같이 보냄 |
| `OUT_DIR` | `work/runs/<UTC 시각>` | 결과 폴더. 이미 파일이 있는 폴더는 거부한다 |

주의할 점

- Host의 QAIRT와 보드 QNN Runtime의 버전을 맞춥니다. 이 결과는 둘 다 2.35입니다.
- 그래프 입출력은 W8A8 양자화된 uint8입니다(v0, v1은 NCHW `[1,256,32,32]`, v2는 NHWC `[1,32,32,256]`). 러너가 float32 ↔ uint8 변환을 하며, 이 변환은 측정 시간에 들어가지 않습니다.
- 측정 조건(HTP 전력 burst, 워밍업 100회 + 측정 300회 × 10 trial, Optrace 21회)은 [board.sh](tools/board.sh)에 있습니다.
- 측정 러너는 `taskset -c 7`로 큰 CPU 코어(QCS6490 기준)에 고정됩니다. 다른 SoC면 [board.sh](tools/board.sh)의 코어 번호를 바꿉니다.
- 측정 중 하나라도 실패하면(러너 종료 코드, 출력 행 수 불일치) 보드 스크립트가 멈추고 `out/logs/`에 로그를 남깁니다.
