// [보드] HTP Context 하나를 로드해 동기 graphExecute 시간을 잰다.
//
// 파일 전체 역할
// 1. Host에서 미리 만든 Context Binary를 읽고, 그 안의 그래프 이름과 입출력 텐서 정보를 꺼낸다
// 2. QNN API 순서대로 Backend -> Device -> Context -> Graph를 준비한다
// 3. float32 입력 파일을 uint8로 양자화해 입력 버퍼에 넣는다 (측정 시간 밖)
// 4. graphExecute를 (워밍업 + 측정) 횟수만큼 반복하며 1회 시간을 잰다
// 5. basic profile을 켜고 20회 더 실행해 HTP 쪽 시간 이벤트를 받는다
// 6. 마지막 출력을 float32로 되돌려 파일로 저장한다 (정확도 확인)
//
// 사용법: bench_graph <context.bin> <input_f32.raw> <warmups> <repeats> <output_f32.raw>
// stdout 출력 형식 (board.sh가 이 형식만 골라 CSV로 모은다)
//   "exec,<us>"         graphExecute 1회 시간 (워밍업 제외, steady_clock)
//   "htp,<us>"          QNN basic profile: Accelerator (execute) time        -> HTP 자원 확보 + 그래프 실행 + 자원 반환
//   "htp_rpc,<us>"      QNN basic profile: QNN accelerator (execute) time    -> HTP 쪽 실행 준비부터 반환까지
//   "host_rpc,<us>"     QNN basic profile: RPC (execute) time                -> FastRPC 요청부터 응답까지
//   "qnn_execute,<us>"  QNN basic profile: QNN (execute) time                -> graphExecute 전체 (exec와 같은 구간을 QNN이 잰 값)
// 이 실습의 그래프는 입출력이 하나씩이고 UFIXED_POINT_8 + scale/offset 양자화
// 측정 조건을 고정하려고 HTP 전력을 burst(최대 클럭, DCVS 끔, sleep 끔)로 설정

#include <HTP/QnnHtpDevice.h>
#include <QnnInterface.h>
#include <System/QnnSystemInterface.h>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// QNN API는 실패해도 예외 없이 오류 코드만 돌려준다
// 모든 QNN 호출을 감싸서, 실패하면 호출 이름을 담은 예외로 바꿈
#define CHECK(x) do { if ((x) != QNN_SUCCESS) throw std::runtime_error(#x); } while (0)

// 조건이 거짓이면 메시지를 담은 예외 발생 (인자, 파일, 텐서 형식 확인에 사용)
static void require(bool ok, const std::string& msg) { if (!ok) throw std::runtime_error(msg); }

// 파일 전체를 바이트 배열로 읽어 반환 (Context Binary, 입력 raw 파일)
static std::vector<char> slurp(const char* p) {
  std::ifstream f(p, std::ios::binary);                 // 바이너리 모드로 열기
  require(bool(f), std::string("cannot open ") + p);    // 파일이 없으면 오류
  return {std::istreambuf_iterator<char>(f), {}};       // 처음부터 끝까지 읽어 vector로
}


static size_t check_tensor(const Qnn_Tensor_t& t, const char* role) {
  // 입출력 텐서가 이 러너가 처리할 수 있는 형식인지 확인하고 원소 수 반환
  // uint8 + 텐서 하나에 scale/offset 한 쌍 + flat 배치 + 4차원이 아니면 양자화 계산이 틀어지므로 바로 오류
  const std::string who = std::string(role) + " tensor: ";
  require(t.version == QNN_TENSOR_VERSION_1, who + "expected Qnn_Tensor_t version 1");

  const auto& v = t.v1;
  require(v.dataType == QNN_DATATYPE_UFIXED_POINT_8, who + "expected QNN_DATATYPE_UFIXED_POINT_8");   // uint8 양자화
  require(v.quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED &&                            // 텐서 전체에 scale/offset 한 쌍
          v.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
          who + "expected per-tensor scale/offset quantization");

  const float scale = v.quantizeParams.scaleOffsetEncoding.scale;
  require(std::isfinite(scale) && scale > 0.f, who + "invalid scale");                                // scale이 NaN, 0 이하면 양자화 불가
  require(v.dataFormat == QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, who + "expected flat buffer layout");   // 일반 연속 배열
  require(v.rank == 4 && v.dimensions != nullptr, who + "expected rank 4");

  size_t n = 1;
  for (uint32_t i = 0; i < v.rank; ++i) { require(v.dimensions[i] > 0, who + "zero dimension"); n *= v.dimensions[i]; }   // 원소 수

  // 텐서 이름, shape, scale/offset을 stderr에 기록 (board.sh가 로그 파일로 남김)
  fprintf(stderr, "%s %s [%u,%u,%u,%u] uint8 scale=%g offset=%d\n", role, v.name, v.dimensions[0], v.dimensions[1],
          v.dimensions[2], v.dimensions[3], scale, v.quantizeParams.scaleOffsetEncoding.offset);

  return n;
}

// main 전체를 try로 -> 어디서든 예외가 나면 맨 끝 catch에서 메시지 출력 후 종료 코드 1
// board.sh는 종료 코드로 실패한 trial을 걸러낸다
int main(int argc, char** argv) try {
  // 1. 인자 확인과 파일 읽기
  require(argc == 6, "usage: bench_graph context.bin input_f32.raw warmups repeats output_f32.raw");   // 프로그램 이름 + 인자 5개
  const int warm = std::stoi(argv[3]), reps = std::stoi(argv[4]);   // 워밍업 횟수, 측정 횟수
  require(warm >= 0 && reps > 0, "warmups >= 0, repeats > 0");
  auto bin = slurp(argv[1]), in_f = slurp(argv[2]);                 // Context Binary, float32 입력 파일


  // 2. QNN 라이브러리 열기
  // libQnnHtp.so는 HTP Backend, libQnnSystem.so는 Context Binary 안의 그래프 정보를 읽는 보조 API
  void* htp = dlopen("libQnnHtp.so", RTLD_NOW);       // LD_LIBRARY_PATH(보드 /usr/lib 또는 함께 보낸 Runtime)에서 찾음
  void* sys = dlopen("libQnnSystem.so", RTLD_NOW);
  if (!htp || !sys) { const char* e = dlerror(); throw std::runtime_error(std::string("dlopen: ") + (e ? e : "?")); }

  // getProviders로 API 함수 테이블을 받아, 이후 q.backendCreate(...), s.systemContextCreate(...)처럼 호출
  const QnnInterface_t** prov = nullptr; const QnnSystemInterface_t** sprov = nullptr; uint32_t n = 0, ns = 0;
  CHECK(((decltype(&QnnInterface_getProviders))dlsym(htp, "QnnInterface_getProviders"))(&prov, &n));               // dlsym으로 함수 주소를 찾아 호출
  CHECK(((decltype(&QnnSystemInterface_getProviders))dlsym(sys, "QnnSystemInterface_getProviders"))(&sprov, &ns));
  require(n > 0 && ns > 0, "no QNN interface provider");
  auto q = prov[0]->QNN_INTERFACE_VER_NAME;           // QNN API 함수 테이블
  auto s = sprov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;   // QNN System API 함수 테이블


  // 3. Context Binary에서 그래프 이름과 입출력 텐서 정의 읽기
  // 그래프 이름은 7의 graphRetrieve(Context 안의 그래프를 이름으로 찾기)에, 텐서 정의(shape, scale/offset)는 양자화와 버퍼 크기에 사용
  QnnSystemContext_Handle_t sh; CHECK(s.systemContextCreate(&sh));
  const QnnSystemContext_BinaryInfo_t* info; Qnn_ContextBinarySize_t isz;
  CHECK(s.systemContextGetBinaryInfo(sh, bin.data(), bin.size(), &info, &isz));   // binary를 해석해 정보 구조체를 받음
  require(info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3, "expected context binary info v3");
  require(info->contextBinaryInfoV3.numGraphs == 1, "expected one graph in the context");
  const auto& gi = info->contextBinaryInfoV3.graphs[0];
  require(gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3, "expected graph info v3");
  const auto& g = gi.graphInfoV3;                                                  // 그래프 이름, 입출력 텐서 정의
  require(g.numGraphInputs == 1 && g.numGraphOutputs == 1, "expected one input and one output");
  Qnn_Tensor_t in = g.graphInputs[0], out = g.graphOutputs[0];                     // 텐서 정의를 복사해 두고, 7에서 버퍼만 붙여 graphExecute에 넘김
  const size_t n_in = check_tensor(in, "input"), n_out = check_tensor(out, "output");
  const auto qi = in.v1.quantizeParams.scaleOffsetEncoding, qo = out.v1.quantizeParams.scaleOffsetEncoding;   // 입력, 출력 scale/offset


  // 4. float32 입력 -> uint8 양자화 (측정 시간 밖)
  require(in_f.size() == n_in * sizeof(float), "input file size != input tensor elements x 4 bytes");   // 파일 크기 = 원소 수 x 4 byte
  std::vector<uint8_t> ib(n_in), ob(n_out);                                         // 입력, 출력 버퍼 (uint8)
  const float* xf = reinterpret_cast<const float*>(in_f.data());                    // 파일 바이트를 float 배열로 해석
  for (size_t i = 0; i < ib.size(); ++i)                                            // q = round(x / scale) - offset
    ib[i] = uint8_t(std::fmin(255.0, std::fmax(0.0, std::nearbyint(xf[i] / qi.scale) - qi.offset)));


  // 5. Backend, Device 생성
  // Backend: libQnnHtp.so 초기화 (대부분의 QNN API는 Backend가 먼저 있어야 함)
  // Device: 실행에 쓸 HTP 자원을 가리키는 handle (6의 전력 설정은 Device infrastructure로 한다)
  Qnn_LogHandle_t log; Qnn_BackendHandle_t backend; Qnn_DeviceHandle_t device;
  CHECK(q.logCreate([](const char* f, QnnLog_Level_t, uint64_t, va_list a) { vfprintf(stderr, f, a); }, QNN_LOG_LEVEL_ERROR, &log));   // QNN 로그는 ERROR만 stderr로
  CHECK(q.backendCreate(log, nullptr, &backend));      // 설정은 기본값
  CHECK(q.deviceCreate(nullptr, nullptr, &device));    // 기본 Device(HTP)


  // 6. 전력 burst 설정 (qnn-net-run --perf_profile burst와 같은 기능)
  // 기본 정책은 부하에 따라 클럭을 바꿔 측정값이 흔들리므로, DCVS를 끄고 전압(클럭)을 최대로 고정
  QnnDevice_Infrastructure_t infra; CHECK(q.deviceGetInfrastructure(&infra));
  auto perf = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra)->perfInfra;     // HTP 전력 설정 인터페이스
  uint32_t pid; CHECK(perf.createPowerConfigId(0, 0, &pid));                          // 전력 설정을 묶을 ID (device 0, core 0)
  QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;               // DCVS v3 방식의 클럭/전압 설정
  auto& d = dcvs.dcvsV3Config;
  d.contextId = pid; d.setDcvsEnable = 1; d.dcvsEnable = 0;                           // DCVS(부하에 따른 클럭 자동 조절) 끔
  d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;               // 성능 우선
  d.setSleepDisable = 1; d.sleepDisable = 1;                                          // 실행 사이에 sleep 들어가지 않게
  d.setBusParams = d.setCoreParams = 1;                                               // 아래 버스, 코어 전압 값을 적용
  d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;     // 버스(메모리) 전압 최대
  d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;  // 코어 전압 최대 -> 최대 클럭
  const QnnHtpPerfInfrastructure_PowerConfig_t* cfg[] = {&dcvs, nullptr};             // 설정 목록 (nullptr로 끝 표시)
  CHECK(perf.setPowerConfig(pid, cfg));


  // 7. Context 로드, Graph 찾기, 입출력 버퍼 연결
  // Context는 Host에서 컴파일이 끝난 binary를 로드만 함 (보드에서는 컴파일하지 않음)
  Qnn_ContextHandle_t ctx; Qnn_GraphHandle_t graph;
  CHECK(q.contextCreateFromBinary(backend, device, nullptr, bin.data(), bin.size(), &ctx, nullptr));   // binary로 Context 생성
  CHECK(q.graphRetrieve(ctx, g.graphName, &graph));                                                    // Context 안의 그래프를 이름("v0_naive" 등)으로 찾아 Graph handle 획득
  in.v1.memType = out.v1.memType = QNN_TENSORMEMTYPE_RAW;    // 앱이 할당한 일반 메모리
  in.v1.clientBuf = {ib.data(), uint32_t(ib.size())};        // {버퍼 주소, 바이트 수}
  out.v1.clientBuf = {ob.data(), uint32_t(ob.size())};


  // 8. 시간 측정
  // graphExecute는 동기 호출이라 반환 시점에 HTP 실행과 출력 쓰기가 끝나 있음 -> 호출 앞뒤 시각 차이가 1회 시간
  std::vector<double> t;
  for (int i = 0; i < warm + reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    CHECK(q.graphExecute(graph, &in, 1, &out, 1, nullptr, nullptr));                          // profile 없이 실행
    auto t1 = std::chrono::steady_clock::now();
    if (i >= warm) t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());   // 처음 warm회는 버리고 µs로 저장
  }
  for (double us : t) printf("exec,%.1f\n", us);   // 측정이 끝난 뒤 한꺼번에 출력 (printf 비용이 측정에 섞이지 않게)


  // 9. basic profile 20회
  // 8과 따로 실행: profile 처리 비용이 exec 측정에 섞이지 않게
  const std::map<std::string, const char*> events = {                                           // QNN 이벤트 이름 -> 출력 이름
      {"Accelerator (execute) time", "htp"}, {"QNN accelerator (execute) time", "htp_rpc"},
      {"RPC (execute) time", "host_rpc"}, {"QNN (execute) time", "qnn_execute"}};
  for (int i = 0; i < 20; ++i) {
    Qnn_ProfileHandle_t prof; CHECK(q.profileCreate(backend, QNN_PROFILE_LEVEL_BASIC, &prof));  // 실행마다 새 profile handle
    CHECK(q.graphExecute(graph, &in, 1, &out, 1, prof, nullptr));                               // profile handle을 붙여 실행
    const QnnProfile_EventId_t* ev; uint32_t ne; CHECK(q.profileGetEvents(prof, &ev, &ne));     // 기록된 이벤트 목록
    size_t found = 0;
    for (uint32_t k = 0; k < ne; ++k) {
      QnnProfile_EventData_t e; CHECK(q.profileGetEventData(ev[k], &e));                        // 이벤트 이름, 값, 단위
      auto it = events.find(e.identifier ? e.identifier : "");
      if (it == events.end()) continue;                                                         // 필요한 네 이벤트만
      require(e.unit == QNN_PROFILE_EVENTUNIT_MICROSEC, std::string("unexpected unit for ") + it->first);
      printf("%s,%llu\n", it->second, (unsigned long long)e.value);                             // "htp,160" 형식으로 출력
      ++found;
    }
    require(found == events.size(), "missing basic profile events");                            // 네 이벤트가 모두 있어야 정상
    CHECK(q.profileFree(prof));
  }


  // 10. 출력 저장과 해제
  std::vector<float> yf(ob.size());
  for (size_t i = 0; i < ob.size(); ++i) yf[i] = float((ob[i] + qo.offset) * qo.scale);           // uint8 -> float: x = (q + offset) * scale
  std::ofstream of(argv[5], std::ios::binary);
  of.write(reinterpret_cast<char*>(yf.data()), yf.size() * 4);                                    // float32 raw로 저장 (정확도 확인용)
  require(bool(of), "cannot write output file");
  CHECK(q.contextFree(ctx, nullptr)); CHECK(q.deviceFree(device)); CHECK(q.backendFree(backend)); // 만든 순서의 반대로 해제
  CHECK(q.logFree(log));
  CHECK(s.systemContextFree(sh));
  return 0;
} catch (const std::exception& e) { fprintf(stderr, "ERROR %s\n", e.what()); return 1; }
