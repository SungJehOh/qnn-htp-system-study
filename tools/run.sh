#!/usr/bin/env bash
# [Host] 모델 생성 -> HTP Context -> 보드 측정 -> Optrace 디코딩 -> 표와 그래프
# source <QAIRT>/bin/envsetup.sh && BOARD=root@<보드 IP> bash tools/run.sh   (환경변수는 README 참고)
#
# 파일 전체 역할
# 1. make_models.py로 ONNX 3개와 입력 데이터 생성
# 2. 그래프마다 QNN 변환(W8A8) -> x86 모델 라이브러리 -> HTP Context Binary 2개(일반, Optrace) 생성
# 3. 보드용 측정 러너(bench_graph) 교차 컴파일
# 4. 보드로 복사해 board.sh 실행, 결과 회수
# 5. Optrace 로그를 schematic과 함께 디코딩해 Chrome trace, QHAS, runtrace 생성
# 6. analyze.py로 결과 표와 타임라인 그림 생성

set -euo pipefail   # 명령 실패, 정의 안 된 변수, 파이프 중간 실패 시 바로 종료


# 경로와 설정
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; WORK="$ROOT/work"                   # 저장소 최상위, 중간 파일 폴더
RES="${OUT_DIR:-$WORK/runs/$(date -u +%Y%m%dT%H%M%SZ)}"                       # 결과 폴더 (기본: 실행 시각)
if [[ -e "$RES" && -n "$(ls -A "$RES" 2>/dev/null)" ]]; then                  # 이미 파일이 있으면 덮어쓰지 않고 종료
  echo "[ERROR] $RES 에 이미 파일이 있다. 결과를 덮어쓰지 않도록 비어 있는 OUT_DIR를 지정한다." >&2; exit 2
fi
SDK="${QNN_SDK_ROOT:?source <QAIRT>/bin/envsetup.sh}"; BIN="$SDK/bin/x86_64-linux-clang"; LIB="$SDK/lib/x86_64-linux-clang"   # Host용 QNN 도구, 라이브러리
BOARD="${BOARD:-rubikpi}"; BOARD_DIR="${BOARD_DIR:-/home/root/qnn-htp-system-study}"   # ssh 대상, 보드 작업 폴더
SOC_ID="${SOC_ID:-35}"; DSP_ARCH="${DSP_ARCH:-v68}"                                    # QCS6490
VARIANTS="v0_naive v1_conv1x1 v2_nhwc"
rm -rf "$WORK/models" "$WORK/qnn" "$WORK/stage" "$WORK/board" && mkdir -p "$WORK/stage/ctx"   # 이전 중간 파일 삭제 (work/ 내부만)
cd "$WORK"   # 변환기가 현재 폴더에 빈 output/ 폴더를 만들므로 작업 폴더에서 실행


# 1. ONNX 생성
echo "== 1. ONNX 생성"
python3 "$ROOT/tools/make_models.py" "$WORK/models"   # work/models/에 ONNX 3개, 입력, calibration 목록


# 2. QNN 변환과 Context 생성
# 그래프마다 ONNX -> QNN 그래프 -> x86 모델 라이브러리 -> HTP Context Binary (일반, Optrace)
echo "== 2. ONNX -> QNN (W8A8) -> HTP $DSP_ARCH Context"

# HTP Backend 설정: vtcm_mb 0 = 사용 가능한 VTCM 최대, O 3 = 그래프 최적화 수준, devices = 컴파일 대상 SoC
cat > "$WORK/htp.json" <<J
{"graphs": [{"graph_names": ["v0_naive", "v1_conv1x1", "v2_nhwc"], "vtcm_mb": 0, "O": 3}],
 "devices": [{"soc_id": $SOC_ID, "dsp_arch": "$DSP_ARCH"}]}
J

# Context 생성기가 위 설정 파일을 읽도록 연결하는 확장 설정
echo "{\"backend_extensions\": {\"shared_library_path\": \"$LIB/libQnnHtpNetRunExtensions.so\", \"config_file_path\": \"$WORK/htp.json\"}}" > "$WORK/ext.json"
for v in $VARIANTS; do
  O="$WORK/qnn/$v"; mkdir -p "$O"

  # v0, v1은 NCHW로 유지 -> HTP 내부 배치와 맞추려고 변환기가 입력 뒤/출력 앞에 Transpose를 넣음
  # v2는 입력이 NHWC라고 알려 줌 -> ONNX 앞뒤의 Transpose가 사라짐
  if [[ $v == v2_nhwc ]]; then IO=(--input_layout x NHWC); L=nhwc; else IO=(--preserve_io layout); L=nchw; fi

  # ONNX -> QNN 그래프(.cpp) + 가중치(.bin), calibration 입력 8개로 W8A8 양자화
  "$BIN/qnn-onnx-converter" --input_network "$WORK/models/$v.onnx" --input_list "$WORK/models/calib_$L.txt" \
    --act_bitwidth 8 --weights_bitwidth 8 "${IO[@]}" --output_path "$O/$v.cpp" > "$O/converter.log" 2>&1

  # Context 생성기가 Host에서 그래프를 구성할 때 쓰는 x86 모델 라이브러리 (lib<v>.so)
  "$BIN/qnn-model-lib-generator" -c "$O/$v.cpp" -b "$O/$v.bin" -t x86_64-linux-clang -l "$v" -o "$O/lib" > "$O/libgen.log" 2>&1

  # HTP용 Context Binary 2개: 일반(시간 측정용), Optrace(계측 포함 + schematic, 디코딩에 필요)
  for kind in plain optrace; do
    P=(); [[ $kind == optrace ]] && P=(--profiling_level detailed --profiling_option optrace)   # Optrace일 때만 옵션 추가
    mkdir -p "$O/$kind"                                                                         # schematic은 현재 폴더에 생기므로 출력 폴더에서 실행
    (cd "$O/$kind" && "$BIN/qnn-context-binary-generator" --model "$O/lib/x86_64-linux-clang/lib$v.so" --backend "$LIB/libQnnHtp.so" \
      --config_file "$WORK/ext.json" "${P[@]}" --output_dir "$O/$kind" --binary_file "$v" > "$O/ctx_$kind.log" 2>&1)
  done
  cp "$O/plain/$v.bin" "$WORK/stage/ctx/$v.bin"; cp "$O/optrace/$v.bin" "$WORK/stage/ctx/${v}_optrace.bin"   # 보드로 보낼 Context 모음

  # 변환된 QNN 그래프의 op 종류별 개수 출력
  python3 -c "import json,collections,sys; g=json.load(open(sys.argv[1]))['graph']; print('  $v QNN ops:', dict(collections.Counter(n['type'] for n in g['nodes'].values())))" "$O/${v}_net.json"
done


# 3. 보드용 측정 러너 빌드
echo "== 3. 보드용 측정 러너 (aarch64 교차 컴파일)"
if [[ -n "${ESDK_ROOT:-}" ]]; then   # 보드 eSDK가 있으면 사용
  ( unset LD_LIBRARY_PATH; source "$ESDK_ROOT"/environment-setup-* > /dev/null
    $CXX -std=c++17 -O2 -I"$SDK/include/QNN" "$ROOT/tools/bench_graph.cpp" -ldl -o "$WORK/stage/bench_graph" )
else                                 # 없으면 일반 aarch64 크로스 컴파일러 (apt install g++-aarch64-linux-gnu)
  "${AARCH64_CXX:-aarch64-linux-gnu-g++}" -std=c++17 -O2 -I"$SDK/include/QNN" "$ROOT/tools/bench_graph.cpp" -ldl \
    -o "$WORK/stage/bench_graph"
fi

# 보드에 QNN Runtime이 없으면 QAIRT의 보드용 Runtime을 같이 보냄
if [[ "${COPY_QNN_RUNTIME:-0}" == 1 ]]; then
  A="${DSP_ARCH^}"; T="${QNN_TARGET:-aarch64-oe-linux-gcc11.2}"; mkdir -p "$WORK/stage/qnn"
  cp "$SDK/bin/$T/qnn-net-run" "$SDK/lib/$T"/libQnn{Htp,System,Htp${A}Stub,HtpNetRunExtensions}.so \
     "$SDK/lib/hexagon-$DSP_ARCH/unsigned"/libQnnHtp${A}{,Skel}.so "$WORK/stage/qnn/"         # CPU 쪽 라이브러리 + cDSP 쪽 Skel
fi


# 4. 보드 실행
echo "== 4. 보드 실행 ($BOARD:$BOARD_DIR)"
cp -r "$WORK/models/inputs" "$ROOT/tools/board.sh" "$WORK/stage/"   # 입력과 보드 스크립트도 stage에
ssh "$BOARD" "rm -rf $BOARD_DIR && mkdir -p $BOARD_DIR"             # 보드 작업 폴더를 새로 만듦
scp -rq "$WORK"/stage/* "$BOARD:$BOARD_DIR/"                        # Context, 입력, 러너, board.sh 복사
ssh "$BOARD" "sh $BOARD_DIR/board.sh"                               # 보드에서 측정
scp -rq "$BOARD:$BOARD_DIR/out" "$WORK/board"                       # 결과 회수


# 5. Optrace 디코딩과 결과 정리
# Optrace 로그는 같은 Context를 만들 때 생긴 schematic이 있어야 HTP op 단위로 풀린다
echo "== 5. Optrace 디코딩 (로그 + Context를 만들 때 생긴 schematic) -> $RES"
mkdir -p "$RES/latency" "$RES/outputs"
cp "$WORK/models/inputs/test_ref_nhwc.raw" "$RES/outputs/"                        # SQNR 계산용 FP32 기준 출력
echo '{"features": {"qhas_json": true, "runtrace": true}}' > "$WORK/reader.json"  # QHAS JSON, runtrace JSON도 생성
for v in $VARIANTS; do
  cp "$WORK/board/$v.csv" "$RES/latency/"                                         # 시간 측정 CSV
  cp "$WORK/board/$v.raw" "$RES/outputs/"                                         # 보드 출력 (float32)
  D="$WORK/board/optrace_$v"
  (cd "$D" && ulimit -s 65536 && "$BIN/qnn-profile-viewer" --reader "$LIB/libQnnHtpOptraceProfilingReader.so" \
    --input_log qnn-profiling-data_0.log --schematic "$WORK/qnn/$v/optrace/${v}_schematic.bin" \
    --config "$WORK/reader.json" --output optrace.json > viewer.log 2>&1) \
    || { echo "[ERROR] qnn-profile-viewer 실패: $D/viewer.log" >&2; exit 1; }       # ulimit -s: 큰 로그 디코딩용 스택
  mkdir -p "$RES/optrace/$v"
  cp "$D/optrace.json" "$RES/optrace/$v/optrace.json"                              # Chrome trace (Perfetto)
  cp "$D/optrace_qnn_htp_analysis_summary.json" "$RES/optrace/$v/qhas.json"        # HTP op별 cycle, 스레드 (첫 실행)
  cp "$D/optrace_runtrace.json" "$RES/optrace/$v/runtrace.json"                    # 21회 실행 각각의 Inference cycle
done


# 6. 결과 표와 그림
echo "== 6. 결과 표와 타임라인 그래프"
python3 "$ROOT/tools/analyze.py" "$RES"   # 결과 표 출력, $RES/figures/optrace_timeline.svg 생성
