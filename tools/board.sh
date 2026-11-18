#!/bin/sh
# [보드] run.sh가 보낸 ctx/ inputs/ bench_graph 로 시간 측정과 Optrace를 실행한다.
#
# 파일 전체 역할
# 1. QNN Runtime 라이브러리 경로 설정 (보드 OS에 있는 것 또는 run.sh가 같이 보낸 것)
# 2. v0, v1, v2를 번갈아 bench_graph로 실행해 graphExecute 시간과 basic profile 값 수집
# 3. qnn-net-run으로 21회씩 실행해 Optrace 로그 생성
# 결과는 out/ 아래: <v>.csv (시간), <v>.raw (마지막 출력), optrace_<v>/ (Optrace 로그), logs/ (실행 로그)

set -eu                 # 명령이 실패하거나 정의 안 된 변수를 쓰면 바로 종료
cd "$(dirname "$0")"    # 스크립트가 있는 폴더로 이동


# QNN Runtime 라이브러리 경로
# QNN은 CPU 쪽(libQnnHtp, Stub)과 cDSP 쪽(Skel) 라이브러리로 나뉘어 경로를 두 개 지정
if [ -d qnn ]; then   # run.sh가 QAIRT Runtime을 같이 보낸 경우
  export PATH="$PWD/qnn:$PATH" LD_LIBRARY_PATH="$PWD/qnn" ADSP_LIBRARY_PATH="$PWD/qnn;/usr/lib/rfsa/adsp"
else                  # 보드 OS에 들어 있는 QNN Runtime (Qualcomm Linux)
  export LD_LIBRARY_PATH=/usr/lib ADSP_LIBRARY_PATH='/usr/lib/rfsa/adsp;/usr/lib'   # CPU 쪽 / cDSP가 Skel을 찾는 경로
fi


# 측정 설정
VARIANTS="v0_naive v1_conv1x1 v2_nhwc"
TRIALS=10 WARMUP=100 REPS=300 PROFILE_RUNS=20             # PROFILE_RUNS는 bench_graph.cpp의 basic profile 횟수와 같아야 함
lay() { [ "$1" = v2_nhwc ] && echo nhwc || echo nchw; }   # 그래프 입력 배치 (v2만 NHWC)
fail() { echo "[board] FAIL: $*" >&2; exit 1; }           # 오류 메시지 출력 후 종료
rm -rf out && mkdir -p out/logs


# 1. 시간 측정
# trial마다 v0 -> v1 -> v2를 새 프로세스로 번갈아 실행 (프로세스 간 변동과 순서 효과를 줄이기 위해)
t=0
while [ $t -lt $TRIALS ]; do
  for v in $VARIANTS; do
    raw=out/logs/${v}_trial$t.stdout err=out/logs/${v}_trial$t.stderr

    # 큰 코어(cpu7, QCS6490 기준)에 고정
    # 파이프로 바로 거르면 종료 코드가 사라지므로 먼저 파일로 받고 종료 코드 확인
    if ! taskset -c 7 ./bench_graph ctx/$v.bin inputs/test_$(lay $v).raw $WARMUP $REPS out/$v.raw > "$raw" 2> "$err"; then
      fail "$v trial $t: bench_graph exit code != 0 (see $err)"
    fi

    # stdout에는 QNN/FastRPC 라이브러리 메시지도 섞이므로 "<kind>,<숫자>" 형식만 골라냄
    grep -E '^(exec|htp|htp_rpc|host_rpc|qnn_execute),[0-9]+(\.[0-9]+)?$' "$raw" > "$raw.csv" || true
    n_exec=$(grep -c '^exec,' "$raw.csv" || true)                                            # 측정 행 수
    n_htp=$(grep -c '^htp,' "$raw.csv" || true)                                              # profile 행 수
    [ "$n_exec" -eq $REPS ] || fail "$v trial $t: exec rows $n_exec != $REPS"
    [ "$n_htp" -eq $PROFILE_RUNS ] || fail "$v trial $t: htp rows $n_htp != $PROFILE_RUNS"

    sed "s/^/$t,/" "$raw.csv" >> out/$v.csv   # 검사를 통과한 trial만 "trial,kind,us"로 기록
  done
  t=$((t + 1))
done


# 2. Optrace
# Optrace Context로 21회 실행 (첫 1회는 워밍업으로 보고 분석에서 뺀다)
# 보드는 21회 모두 op 단위로 기록하지만, qnn-profile-viewer는 op 단위 상세(Chrome trace, QHAS)를 첫 실행만 풀어 준다.
# runtrace만 21회 모두 나온다.
echo '{"devices":[{"cores":[{"perf_profile":"burst"}]}]}' > htp.json   # HTP 설정: 전력 burst

# --perf_profile은 NetRun extension을 config로 로드해야 실제로 적용됨
echo '{"backend_extensions":{"shared_library_path":"libQnnHtpNetRunExtensions.so","config_file_path":"htp.json"}}' > netrun.json
for v in $VARIANTS; do
  echo "inputs/test_$(lay $v).raw" > list.txt   # qnn-net-run 입력 목록 (한 줄에 파일 하나)
  qnn-net-run --backend libQnnHtp.so --retrieve_context ctx/${v}_optrace.bin --input_list list.txt \
    --output_dir out/optrace_$v --profiling_level detailed --profiling_option optrace \
    --perf_profile burst --config_file netrun.json --num_inferences 21 > out/logs/${v}_optrace.log 2>&1 \
    || fail "$v: qnn-net-run optrace failed (see out/logs/${v}_optrace.log)"
  [ -s out/optrace_$v/qnn-profiling-data_0.log ] || fail "$v: no optrace profiling log"   # Optrace 로그가 생겼는지 확인
done
echo "[board] done"
