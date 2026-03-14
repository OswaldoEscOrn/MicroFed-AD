#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <array>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>

#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_task_wdt.h"
#include "esp_vfs.h"

#include "driver/uart.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

#include "custom_tile.h"

static const char *TAG = "ONLINE_EVAL";

#ifndef MODEL_FILENAME
#define MODEL_FILENAME "/spiffs/conv_100k_model.tflite"
#endif

#ifndef RUN_EVERY_MINUTES
#define RUN_EVERY_MINUTES 10
#endif

#ifndef MINUTE_SAMPLE_PERIOD_SEC
#define MINUTE_SAMPLE_PERIOD_SEC 60
#endif

#ifndef NOISE_DB_TH
#define NOISE_DB_TH 65.0f
#endif

#ifndef REFRACTORY_SEC
#define REFRACTORY_SEC 5
#endif

#ifndef MAE_THRESHOLD
// CONV 100k + 100% real -> 0.078524f
// LSTM 100k + 100% real -> 0.36328f
#define MAE_THRESHOLD 0.078524f
#endif

#ifndef LOG_FILENAME
#define LOG_FILENAME "/spiffs/infer_log.csv"
#endif

#ifndef AUTO_DUMP_LOG_ON_BOOT
#define AUTO_DUMP_LOG_ON_BOOT 1
#endif

#ifndef CLEAR_LOG_ON_BOOT
#define CLEAR_LOG_ON_BOOT 0
#endif

#ifndef ENABLE_UART_COMMANDS
#define ENABLE_UART_COMMANDS 1
#endif

static constexpr int kTimesteps = 24;
static constexpr int kFeatures = 4;
static constexpr size_t kTensorArenaBytes = 8 * 1024 * 1024;

// ===== scaler parameters =====
static constexpr float PM25_RAW_MEAN = 80.2259881931f;
static constexpr float PM25_RAW_STD = 76.1980943132f;

static constexpr float DURATION_MEAN = 0.89847821f;
static constexpr float DURATION_STD = 1.86383191f;

static constexpr float EVENT_MEAN = 0.24903034f;
static constexpr float EVENT_STD = 0.49798439f;

static constexpr float SALIENCE_MEAN = 0.29744268f;
static constexpr float SALIENCE_STD = 0.59918661f;

// -------------------- helpers --------------------
template <typename T>
static inline T RegValue(T v) { return v; }

template <typename T>
static inline T RegValue(const T *p) { return *p; }

static float safe_div(float a, float b, float fallback = 0.0f)
{
  return (std::fabs(b) < 1e-8f) ? fallback : (a / b);
}

static float zscore(float x, float mean, float std)
{
  return safe_div(x - mean, std, 0.0f);
}

static void list_spiffs(const char *path)
{
  DIR *dir = opendir(path);
  if (!dir)
  {
    ESP_LOGW(TAG, "Failed to open dir %s", path);
    return;
  }
  dirent *ent;
  while ((ent = readdir(dir)) != nullptr)
  {
    ESP_LOGI(TAG, "  %s", ent->d_name);
  }
  closedir(dir);
}

static bool file_exists(const char *path)
{
  struct stat st;
  return stat(path, &st) == 0;
}

static bool delete_file_if_exists(const char *path)
{
  if (!file_exists(path))
    return true;
  return std::remove(path) == 0;
}

static bool load_file_to_psram(const char *path, uint8_t **out_buf, size_t *out_size)
{
  FILE *f = fopen(path, "rb");
  if (!f)
  {
    ESP_LOGE(TAG, "Failed to open file: %s (errno=%d)", path, errno);
    return false;
  }

  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  if (sz <= 0)
  {
    fclose(f);
    ESP_LOGE(TAG, "File empty: %s", path);
    return false;
  }
  fseek(f, 0, SEEK_SET);

  uint8_t *buf = static_cast<uint8_t *>(
      heap_caps_malloc(static_cast<size_t>(sz), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!buf)
  {
    fclose(f);
    ESP_LOGE(TAG, "PSRAM malloc failed for %ld bytes", sz);
    return false;
  }

  size_t rd = fread(buf, 1, static_cast<size_t>(sz), f);
  fclose(f);
  if (rd != static_cast<size_t>(sz))
  {
    heap_caps_free(buf);
    ESP_LOGE(TAG, "Short read %u/%ld", static_cast<unsigned>(rd), sz);
    return false;
  }

  *out_buf = buf;
  *out_size = static_cast<size_t>(sz);
  return true;
}

static TfLiteStatus NoOpParse(const tflite::Operator *op,
                              tflite::ErrorReporter *reporter,
                              tflite::BuiltinDataAllocator *allocator,
                              void **builtin_data)
{
  (void)op;
  (void)reporter;
  (void)allocator;
  *builtin_data = nullptr;
  return kTfLiteOk;
}

// -------------------- feature structs --------------------
struct HourFeatureRaw
{
  float avg_pm25 = 0.0f;
  float total_noise_duration_sec = 0.0f;
  float noise_event_count = 0.0f;
  float avg_salience = 0.0f;
};

struct HourAccumulator
{
  float pm25_sum = 0.0f;
  int pm25_count = 0;

  float salience_sum = 0.0f;
  int salience_count = 0;

  int noise_duration_sec = 0;
  int noise_event_count = 0;

  bool prev_above_th = false;
  int refractory_left = 0;

  int elapsed_sec = 0;

  void reset()
  {
    pm25_sum = 0.0f;
    pm25_count = 0;
    salience_sum = 0.0f;
    salience_count = 0;
    noise_duration_sec = 0;
    noise_event_count = 0;
    prev_above_th = false;
    refractory_left = 0;
    elapsed_sec = 0;
  }
};

struct RuntimeState
{
  std::array<HourFeatureRaw, kTimesteps> hour_ring{};
  int hour_count = 0;
  int hour_head = 0;

  HourAccumulator hour_acc{};

  float noise_baseline_db = 45.0f;
  bool baseline_init = false;

  int sec_counter = 0;
  int minute_counter = 0;
  int inference_counter = 0;
};

static RuntimeState g_state;

// -------------------- logging --------------------
static void ensure_log_header()
{
  if (file_exists(LOG_FILENAME))
    return;

  FILE *f = fopen(LOG_FILENAME, "w");
  if (!f)
  {
    ESP_LOGE(TAG, "Failed to create log file: %s", LOG_FILENAME);
    return;
  }

  fprintf(f,
          "minute_counter,"
          "avg_pm25,"
          "duration_hour_equiv,"
          "event_hour_equiv,"
          "avg_salience,"
          "mae,"
          "threshold,"
          "is_anomaly\n");
  fclose(f);

  ESP_LOGI(TAG, "Created log file with header: %s", LOG_FILENAME);
}

static void append_infer_log(int minute_counter,
                             const HourFeatureRaw &partial,
                             float mae,
                             float threshold,
                             bool is_anomaly)
{
  FILE *f = fopen(LOG_FILENAME, "a");
  if (!f)
  {
    ESP_LOGE(TAG, "Failed to append log file: %s", LOG_FILENAME);
    return;
  }

  fprintf(f,
          "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
          minute_counter,
          partial.avg_pm25,
          partial.total_noise_duration_sec,
          partial.noise_event_count,
          partial.avg_salience,
          mae,
          threshold,
          is_anomaly ? 1 : 0);

  fclose(f);
}

static void dump_log_to_serial()
{
  FILE *f = fopen(LOG_FILENAME, "r");
  if (!f)
  {
    ESP_LOGW(TAG, "No log file to dump: %s", LOG_FILENAME);
    return;
  }

  ESP_LOGI(TAG, "===== BEGIN LOG DUMP: %s =====", LOG_FILENAME);

  char line[256];
  while (fgets(line, sizeof(line), f))
  {
    printf("%s", line);
  }

  fclose(f);
  ESP_LOGI(TAG, "===== END LOG DUMP =====");
}

static void clear_log_and_recreate()
{
  if (!delete_file_if_exists(LOG_FILENAME))
  {
    ESP_LOGE(TAG, "Failed to delete log file: %s", LOG_FILENAME);
    return;
  }
  ensure_log_header();
  ESP_LOGI(TAG, "Log cleared");
}

// -------------------- op registration --------------------
static bool register_required_ops(tflite::MicroMutableOpResolver<96> &resolver)
{
  (void)resolver.AddBuiltin(tflite::BuiltinOperator_ADD, RegValue(tflite::Register_ADD()), tflite::ParseAdd);
  (void)resolver.AddBuiltin(tflite::BuiltinOperator_MUL, RegValue(tflite::Register_MUL()), tflite::ParseMul);
  (void)resolver.AddBuiltin(tflite::BuiltinOperator_SUB, RegValue(tflite::Register_SUB()), tflite::ParseSub);
  (void)resolver.AddBuiltin(tflite::BuiltinOperator_DIV, RegValue(tflite::Register_DIV()), tflite::ParseDiv);

  (void)resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE, RegValue(tflite::Register_RESHAPE()), tflite::ParseReshape);
  (void)resolver.AddBuiltin(tflite::BuiltinOperator_EXPAND_DIMS, RegValue(tflite::Register_EXPAND_DIMS()), tflite::ParseExpandDims);
  (void)resolver.AddBuiltin(tflite::BuiltinOperator_STRIDED_SLICE, RegValue(tflite::Register_STRIDED_SLICE()), tflite::ParseStridedSlice);
  (void)resolver.AddBuiltin(tflite::BuiltinOperator_CONCATENATION, RegValue(tflite::Register_CONCATENATION()), tflite::ParseConcatenation);

  (void)resolver.AddBuiltin(tflite::BuiltinOperator_TANH, RegValue(tflite::Register_TANH()), tflite::ParseTanh);
  (void)resolver.AddBuiltin(tflite::BuiltinOperator_LOGISTIC, RegValue(tflite::Register_LOGISTIC()), tflite::ParseLogistic);

  (void)resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                            RegValue(tflite::Register_FULLY_CONNECTED()),
                            tflite::ParseFullyConnected);

  (void)resolver.AddBuiltin(tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
                            RegValue(tflite::Register_UNIDIRECTIONAL_SEQUENCE_LSTM()),
                            tflite::ParseUnidirectionalSequenceLSTM);

  (void)resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                            RegValue(tflite::Register_CONV_2D()),
                            tflite::ParseConv2D);

  (void)resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                            RegValue(tflite::Register_MAX_POOL_2D()),
                            tflite::ParsePool);

  (void)resolver.AddBuiltin(tflite::BuiltinOperator_SHAPE,
                            RegValue(tflite::Register_SHAPE()),
                            NoOpParse);

  (void)resolver.AddBuiltin(tflite::BuiltinOperator_TILE,
                            RegValue(custom::Register_CUSTOM_TILE()),
                            NoOpParse);

  return true;
}

// -------------------- sensor stubs --------------------
// 这里必须替换成你的真实驱动读取。
// 不是手动填数，而是把真实传感器库的读取函数接进来。
static bool read_pm25_sensor(float *out_pm25)
{
  if (!out_pm25)
    return false;

  // TODO: 替换成你真实 PM2.5 传感器读取
  // 例如：*out_pm25 = pms7003_get_pm25();
  *out_pm25 = 80.0f;
  return true;
}

static bool read_noise_db_sensor(float *out_db)
{
  if (!out_db)
    return false;

  // TODO: 替换成你真实噪声 dB 读取/计算
  // 例如：*out_db = microphone_get_db();
  *out_db = 50.0f;
  return true;
}

// -------------------- feature processing --------------------
static void update_noise_baseline(float db)
{
  if (!g_state.baseline_init)
  {
    g_state.noise_baseline_db = db;
    g_state.baseline_init = true;
    return;
  }
  const float alpha = 0.01f;
  g_state.noise_baseline_db = alpha * db + (1.0f - alpha) * g_state.noise_baseline_db;
}

// 如果你的噪声事件定义不同，主要改这个函数
static void process_one_second_sample()
{
  float pm25 = 0.0f;
  float db = 0.0f;

  bool ok_pm25 = read_pm25_sensor(&pm25);
  bool ok_db = read_noise_db_sensor(&db);

  if (ok_pm25)
  {
    g_state.hour_acc.pm25_sum += pm25;
    g_state.hour_acc.pm25_count += 1;
  }

  if (ok_db)
  {
    update_noise_baseline(db);

    const bool above = (db > NOISE_DB_TH);

    if (above)
    {
      g_state.hour_acc.noise_duration_sec += 1;
    }

    if (g_state.hour_acc.refractory_left > 0)
    {
      g_state.hour_acc.refractory_left -= 1;
    }

    if (above && !g_state.hour_acc.prev_above_th && g_state.hour_acc.refractory_left == 0)
    {
      g_state.hour_acc.noise_event_count += 1;
      g_state.hour_acc.refractory_left = REFRACTORY_SEC;
    }

    g_state.hour_acc.prev_above_th = above;

    float salience = std::max(0.0f, db - g_state.noise_baseline_db);
    g_state.hour_acc.salience_sum += salience;
    g_state.hour_acc.salience_count += 1;
  }

  g_state.hour_acc.elapsed_sec += 1;
}

static HourFeatureRaw finalize_completed_hour_feature()
{
  HourFeatureRaw h{};
  h.avg_pm25 = safe_div(g_state.hour_acc.pm25_sum, static_cast<float>(g_state.hour_acc.pm25_count), 0.0f);
  h.total_noise_duration_sec = static_cast<float>(g_state.hour_acc.noise_duration_sec);
  h.noise_event_count = static_cast<float>(g_state.hour_acc.noise_event_count);
  h.avg_salience = safe_div(g_state.hour_acc.salience_sum, static_cast<float>(g_state.hour_acc.salience_count), 0.0f);
  return h;
}

static HourFeatureRaw make_partial_hour_feature()
{
  HourFeatureRaw h{};
  const float elapsed = static_cast<float>(std::max(g_state.hour_acc.elapsed_sec, 1));

  h.avg_pm25 = safe_div(g_state.hour_acc.pm25_sum, static_cast<float>(g_state.hour_acc.pm25_count), 0.0f);
  h.avg_salience = safe_div(g_state.hour_acc.salience_sum, static_cast<float>(g_state.hour_acc.salience_count), 0.0f);

  h.total_noise_duration_sec =
      (static_cast<float>(g_state.hour_acc.noise_duration_sec) / elapsed) * 3600.0f;
  h.noise_event_count =
      (static_cast<float>(g_state.hour_acc.noise_event_count) / elapsed) * 3600.0f;

  return h;
}

static void push_completed_hour(const HourFeatureRaw &h)
{
  g_state.hour_ring[g_state.hour_head] = h;
  g_state.hour_head = (g_state.hour_head + 1) % kTimesteps;
  if (g_state.hour_count < kTimesteps)
    g_state.hour_count++;
}

static float scale_pm25(float avg_pm25_raw)
{
  return zscore(avg_pm25_raw, PM25_RAW_MEAN, PM25_RAW_STD);
}

static float scale_duration(float duration_hour_equiv)
{
  return zscore(duration_hour_equiv, DURATION_MEAN, DURATION_STD);
}

static float scale_event(float event_hour_equiv)
{
  return zscore(event_hour_equiv, EVENT_MEAN, EVENT_STD);
}

static float scale_salience(float avg_salience)
{
  return zscore(avg_salience, SALIENCE_MEAN, SALIENCE_STD);
}

static bool build_model_input(float *dst_24x4)
{
  if (!dst_24x4)
    return false;

  if (g_state.hour_count < (kTimesteps - 1))
  {
    ESP_LOGW(TAG, "Not enough history: have %d complete hours, need %d",
             g_state.hour_count, kTimesteps - 1);
    return false;
  }

  HourFeatureRaw current_partial = make_partial_hour_feature();

  const int completed_to_use = kTimesteps - 1;
  const int completed_start = (g_state.hour_head - completed_to_use + kTimesteps) % kTimesteps;

  int out_idx = 0;
  for (int i = 0; i < completed_to_use; ++i)
  {
    const int idx = (completed_start + i) % kTimesteps;
    const HourFeatureRaw &h = g_state.hour_ring[idx];

    dst_24x4[out_idx++] = scale_pm25(h.avg_pm25);
    dst_24x4[out_idx++] = scale_duration(h.total_noise_duration_sec);
    dst_24x4[out_idx++] = scale_event(h.noise_event_count);
    dst_24x4[out_idx++] = scale_salience(h.avg_salience);
  }

  dst_24x4[out_idx++] = scale_pm25(current_partial.avg_pm25);
  dst_24x4[out_idx++] = scale_duration(current_partial.total_noise_duration_sec);
  dst_24x4[out_idx++] = scale_event(current_partial.noise_event_count);
  dst_24x4[out_idx++] = scale_salience(current_partial.avg_salience);

  return true;
}

static float compute_mae(const float *a, const float *b, size_t n)
{
  float mae = 0.0f;
  for (size_t i = 0; i < n; ++i)
  {
    mae += std::fabs(a[i] - b[i]);
  }
  return (n > 0) ? (mae / static_cast<float>(n)) : 0.0f;
}

// -------------------- uart commands --------------------
#if ENABLE_UART_COMMANDS
static void handle_uart_command_line(const char *line)
{
  if (!line)
    return;

  if (strcmp(line, "dump") == 0)
  {
    dump_log_to_serial();
  }
  else if (strcmp(line, "clear") == 0)
  {
    clear_log_and_recreate();
  }
  else if (strcmp(line, "status") == 0)
  {
    ESP_LOGI(TAG,
             "status | minute_counter=%d hour_count=%d inference_counter=%d baseline_db=%.2f",
             g_state.minute_counter,
             g_state.hour_count,
             g_state.inference_counter,
             static_cast<double>(g_state.noise_baseline_db));
  }
  else if (strlen(line) > 0)
  {
    ESP_LOGI(TAG, "Unknown command: %s", line);
    ESP_LOGI(TAG, "Available: dump | clear | status");
  }
}

static void poll_uart_commands()
{
  char buf[128];
  int len = uart_read_bytes(UART_NUM_0, buf, sizeof(buf) - 1, 0);
  if (len <= 0)
    return;

  buf[len] = '\0';

  char *saveptr = nullptr;
  char *token = strtok_r(buf, "\r\n", &saveptr);
  while (token)
  {
    handle_uart_command_line(token);
    token = strtok_r(nullptr, "\r\n", &saveptr);
  }
}
#endif

// -------------------- app_main --------------------
extern "C" void app_main(void)
{
  esp_task_wdt_config_t cfg = {
      .timeout_ms = 60000,
      .idle_core_mask = (1 << 0) | (1 << 1),
      .trigger_panic = false};
  (void)esp_task_wdt_reconfigure(&cfg);
  ESP_LOGW(TAG, "task_wdt timeout set to %d ms", cfg.timeout_ms);

  ESP_LOGI(TAG, "Starting online inference. Model: %s", MODEL_FILENAME);

  esp_vfs_spiffs_conf_t conf = {
      .base_path = "/spiffs",
      .partition_label = "spiffs",
      .max_files = 8,
      .format_if_mount_failed = false};
  esp_err_t ret = esp_vfs_spiffs_register(&conf);
  if (ret != ESP_OK)
  {
    ESP_LOGE(TAG, "SPIFFS mount failed (err=0x%x)", static_cast<unsigned>(ret));
    return;
  }

  ESP_LOGI(TAG, "SPIFFS mounted successfully");
  ESP_LOGI(TAG, "Files in /spiffs:");
  list_spiffs("/spiffs");

  if (CLEAR_LOG_ON_BOOT)
  {
    clear_log_and_recreate();
  }
  else
  {
    ensure_log_header();
  }

  if (AUTO_DUMP_LOG_ON_BOOT)
  {
    dump_log_to_serial();
  }

  uint8_t *model_buf = nullptr;
  size_t model_size = 0;
  if (!load_file_to_psram(MODEL_FILENAME, &model_buf, &model_size))
    return;

  ESP_LOGI(TAG, "Model loaded (%u bytes) at %p",
           static_cast<unsigned>(model_size), (void *)model_buf);

  const tflite::Model *model = tflite::GetModel(model_buf);
  if (!model || model->version() != TFLITE_SCHEMA_VERSION)
  {
    ESP_LOGE(TAG, "Model schema mismatch");
    return;
  }

  uint8_t *tensor_arena = static_cast<uint8_t *>(
      heap_caps_malloc(kTensorArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!tensor_arena)
  {
    ESP_LOGE(TAG, "tensor arena alloc failed");
    return;
  }

  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter *error_reporter = &micro_error_reporter;
  (void)error_reporter;

  static tflite::MicroMutableOpResolver<96> resolver;
  if (!register_required_ops(resolver))
  {
    ESP_LOGE(TAG, "Op registration failed");
    return;
  }
  ESP_LOGI(TAG, "Operators registered");

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaBytes,
      nullptr, nullptr, true);

  tflite::MicroInterpreter *interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    ESP_LOGE(TAG, "AllocateTensors failed");
    return;
  }

  TfLiteTensor *input = interpreter->input(0);
  TfLiteTensor *output = interpreter->output(0);

  ESP_LOGI(TAG, "TFLM ready. input bytes=%u output bytes=%u arena_used=%u",
           static_cast<unsigned>(input->bytes),
           static_cast<unsigned>(output->bytes),
           static_cast<unsigned>(interpreter->arena_used_bytes()));

  const size_t sample_floats = static_cast<size_t>(kTimesteps) * static_cast<size_t>(kFeatures);
  const size_t sample_bytes = sample_floats * sizeof(float);

  if (input->bytes != sample_bytes || output->bytes != sample_bytes)
  {
    ESP_LOGW(TAG, "Unexpected tensor bytes. expected=%u input=%u output=%u",
             static_cast<unsigned>(sample_bytes),
             static_cast<unsigned>(input->bytes),
             static_cast<unsigned>(output->bytes));
  }

  g_state.hour_acc.reset();

  while (true)
  {
    process_one_second_sample();
    g_state.sec_counter++;

#if ENABLE_UART_COMMANDS
    poll_uart_commands();
#endif

    if (g_state.sec_counter % MINUTE_SAMPLE_PERIOD_SEC == 0)
    {
      g_state.minute_counter++;

      float cur_pm25 = safe_div(g_state.hour_acc.pm25_sum,
                                static_cast<float>(g_state.hour_acc.pm25_count), 0.0f);
      float cur_sal = safe_div(g_state.hour_acc.salience_sum,
                               static_cast<float>(g_state.hour_acc.salience_count), 0.0f);

      ESP_LOGI(TAG,
               "Minute %d | avg_pm25=%.3f duration_so_far=%d sec events_so_far=%d salience_avg=%.3f baseline_db=%.2f",
               g_state.minute_counter,
               static_cast<double>(cur_pm25),
               g_state.hour_acc.noise_duration_sec,
               g_state.hour_acc.noise_event_count,
               static_cast<double>(cur_sal),
               static_cast<double>(g_state.noise_baseline_db));

      if ((g_state.minute_counter % 60) == 0)
      {
        HourFeatureRaw completed = finalize_completed_hour_feature();
        push_completed_hour(completed);

        ESP_LOGI(TAG,
                 "Completed hour pushed | avg_pm25=%.3f duration=%.1f sec events=%.1f salience=%.3f | history=%d/24",
                 static_cast<double>(completed.avg_pm25),
                 static_cast<double>(completed.total_noise_duration_sec),
                 static_cast<double>(completed.noise_event_count),
                 static_cast<double>(completed.avg_salience),
                 g_state.hour_count);

        g_state.hour_acc.reset();
      }

      if ((g_state.minute_counter % RUN_EVERY_MINUTES) == 0)
      {
        float input_window[kTimesteps * kFeatures] = {0};

        if (build_model_input(input_window))
        {
          memcpy(input->data.raw, input_window, sample_bytes);

          uint64_t t0 = esp_timer_get_time();
          TfLiteStatus s = interpreter->Invoke();
          uint64_t dt_us = esp_timer_get_time() - t0;

          if (s != kTfLiteOk)
          {
            ESP_LOGE(TAG, "Invoke failed");
          }
          else
          {
            const float *in_f = reinterpret_cast<const float *>(input->data.raw);
            const float *out_f = reinterpret_cast<const float *>(output->data.raw);

            float mae = compute_mae(out_f, in_f, sample_floats);
            bool is_anomaly = (mae > MAE_THRESHOLD);

            HourFeatureRaw partial = make_partial_hour_feature();
            g_state.inference_counter++;

            ESP_LOGI(TAG,
                     "Inference @ minute=%d | mae=%.6f threshold=%.6f anomaly=%s infer=%.2f ms",
                     g_state.minute_counter,
                     static_cast<double>(mae),
                     static_cast<double>(MAE_THRESHOLD),
                     is_anomaly ? "YES" : "NO",
                     static_cast<double>(dt_us) / 1000.0);

            ESP_LOGI(TAG,
                     "Current partial hour raw | avg_pm25=%.3f duration_eq=%.1f event_eq=%.1f salience=%.3f",
                     static_cast<double>(partial.avg_pm25),
                     static_cast<double>(partial.total_noise_duration_sec),
                     static_cast<double>(partial.noise_event_count),
                     static_cast<double>(partial.avg_salience));

            append_infer_log(g_state.minute_counter, partial, mae, MAE_THRESHOLD, is_anomaly);
          }
        }
      }
    }

    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}
