// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "benchmark/benchmark.h"
#include "benchmark_api_internal.h"
#include "internal_macros.h"

#ifndef BENCHMARK_OS_WINDOWS
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include "check.h"
#include "colorprint.h"
#include "commandlineflags.h"
#include "complexity.h"
#include "log.h"
#include "mutex.h"
#include "re.h"
#include "stat.h"
#include "string_util.h"
#include "sysinfo.h"
#include "timers.h"

DEFINE_bool(benchmark_list_tests, false,
            "Print a list of benchmarks. This option overrides all other "
            "options.");

DEFINE_string(benchmark_filter, ".",
              "A regular expression that specifies the set of benchmarks "
              "to execute.  If this flag is empty, no benchmarks are run.  "
              "If this flag is the string \"all\", all benchmarks linked "
              "into the process are run.");

DEFINE_double(benchmark_min_time, 0.5,
              "Minimum number of seconds we should run benchmark before "
              "results are considered significant.  For cpu-time based "
              "tests, this is the lower bound on the total cpu time "
              "used by all threads that make up the test.  For real-time "
              "based tests, this is the lower bound on the elapsed time "
              "of the benchmark execution, regardless of number of "
              "threads.");

DEFINE_int32(benchmark_repetitions, 1,
             "The number of runs of each benchmark. If greater than 1, the "
             "mean and standard deviation of the runs will be reported.");

DEFINE_bool(benchmark_report_aggregates_only, false,
            "Report the result of each benchmark repetitions. When 'true' is "
            "specified only the mean, standard deviation, and other statistics "
            "are reported for repeated benchmarks.");

DEFINE_string(benchmark_format, "console",
              "The format to use for console output. Valid values are "
              "'console', 'json', or 'csv'.");

DEFINE_string(benchmark_out_format, "json",
              "The format to use for file output. Valid values are "
              "'console', 'json', or 'csv'.");

DEFINE_string(benchmark_out, "", "The file to write additonal output to");

DEFINE_string(benchmark_color, "auto",
              "Whether to use colors in the output.  Valid values: "
              "'true'/'yes'/1, 'false'/'no'/0, and 'auto'. 'auto' means to use "
              "colors if the output is being sent to a terminal and the TERM "
              "environment variable is set to a terminal type that supports "
              "colors.");

DEFINE_int32(v, 0, "The level of verbose logging to output");

namespace benchmark {
namespace internal {

void UseCharPointer(char const volatile*) {}

}  // end namespace internal

namespace {

static const size_t kMaxIterations = 1000000000;

}  // end namespace

namespace internal {

class ThreadManager {
 public:
  ThreadManager(int num_threads)
      : alive_threads_(num_threads), start_stop_barrier_(num_threads) {}

  Mutex& GetBenchmarkMutex() const RETURN_CAPABILITY(benchmark_mutex_) {
    return benchmark_mutex_;
  }

  bool StartStopBarrier() EXCLUDES(end_cond_mutex_) {
    return start_stop_barrier_.wait();
  }

  void NotifyThreadComplete() EXCLUDES(end_cond_mutex_) {
    start_stop_barrier_.removeThread();
    if (--alive_threads_ == 0) {
      MutexLock lock(end_cond_mutex_);
      end_condition_.notify_all();
    }
  }

  void WaitForAllThreads() EXCLUDES(end_cond_mutex_) {
    MutexLock lock(end_cond_mutex_);
    end_condition_.wait(lock.native_handle(),
                        [this]() { return alive_threads_ == 0; });
  }

 public:
  struct Result {
    double real_time_used = 0;
    double cpu_time_used = 0;
    double manual_time_used = 0;
    int64_t bytes_processed = 0;
    int64_t items_processed = 0;
    int complexity_n = 0;
    std::string report_label_;
    std::string error_message_;
    bool has_error_ = false;
  };
  GUARDED_BY(GetBenchmarkMutex()) Result results;

 private:
  mutable Mutex benchmark_mutex_;
  std::atomic<int> alive_threads_;
  Barrier start_stop_barrier_;
  Mutex end_cond_mutex_;
  Condition end_condition_;
};

// Timer management class
class ThreadTimer {
 public:
  ThreadTimer() = default;

  // Called by each thread
  void StartTimer() {
    running_ = true;
    start_real_time_ = ChronoClockNow();
    start_cpu_time_ = ThreadCPUUsage();
  }

  // Called by each thread
  void StopTimer() {
    CHECK(running_);
    running_ = false;
    real_time_used_ += ChronoClockNow() - start_real_time_;
    cpu_time_used_ += ThreadCPUUsage() - start_cpu_time_;
  }

  // Called by each thread
  void SetIterationTime(double seconds) { manual_time_used_ += seconds; }

  bool running() const { return running_; }

  // REQUIRES: timer is not running
  double real_time_used() {
    CHECK(!running_);
    return real_time_used_;
  }

  // REQUIRES: timer is not running
  double cpu_time_used() {
    CHECK(!running_);
    return cpu_time_used_;
  }

  // REQUIRES: timer is not running
  double manual_time_used() {
    CHECK(!running_);
    return manual_time_used_;
  }

 private:
  bool running_ = false;        // Is the timer running
  double start_real_time_ = 0;  // If running_
  double start_cpu_time_ = 0;   // If running_

  // Accumulated time so far (does not contain current slice if running_)
  double real_time_used_ = 0;
  double cpu_time_used_ = 0;
  // Manually set iteration time. User sets this with SetIterationTime(seconds).
  double manual_time_used_ = 0;
};

namespace {

BenchmarkReporter::Run CreateRunReport(
    const benchmark::internal::Benchmark::Instance& b,
    const internal::ThreadManager::Result& results, size_t iters,
    double seconds) {
  // Create report about this benchmark run.
  BenchmarkReporter::Run report;

  report.benchmark_name = b.name;
  report.error_occurred = results.has_error_;
  report.error_message = results.error_message_;
  report.report_label = results.report_label_;
  // Report the total iterations across all threads.
  report.iterations = static_cast<int64_t>(iters) * b.threads;
  report.time_unit = b.time_unit;

  if (!report.error_occurred) {
    double bytes_per_second = 0;
    if (results.bytes_processed > 0 && seconds > 0.0) {
      bytes_per_second = (results.bytes_processed / seconds);
    }
    double items_per_second = 0;
    if (results.items_processed > 0 && seconds > 0.0) {
      items_per_second = (results.items_processed / seconds);
    }

    if (b.use_manual_time) {
      report.real_accumulated_time = results.manual_time_used;
    } else {
      report.real_accumulated_time = results.real_time_used;
    }
    report.cpu_accumulated_time = results.cpu_time_used;
    report.bytes_per_second = bytes_per_second;
    report.items_per_second = items_per_second;
    report.complexity_n = results.complexity_n;
    report.complexity = b.complexity;
    report.complexity_lambda = b.complexity_lambda;
  }
  return report;
}

// Execute one thread of benchmark b for the specified number of iterations.
// Adds the stats collected for the thread into *total.
void RunInThread(const benchmark::internal::Benchmark::Instance* b,
                 size_t iters, int thread_id,
                 internal::ThreadManager* manager) {
  internal::ThreadTimer timer;
  State st(iters, b->arg, thread_id, b->threads, &timer, manager);
  b->benchmark->Run(st);
  CHECK(st.iterations() == st.max_iterations)
      << "Benchmark returned before State::KeepRunning() returned false!";
  {
    MutexLock l(manager->GetBenchmarkMutex());
    internal::ThreadManager::Result& results = manager->results;
    results.cpu_time_used += timer.cpu_time_used();
    results.real_time_used += timer.real_time_used();
    results.manual_time_used += timer.manual_time_used();
    results.bytes_processed += st.bytes_processed();
    results.items_processed += st.items_processed();
    results.complexity_n += st.complexity_length_n();
  }
  manager->NotifyThreadComplete();
}

std::vector<BenchmarkReporter::Run> RunBenchmark(
    const benchmark::internal::Benchmark::Instance& b,
    std::vector<BenchmarkReporter::Run>* complexity_reports) {
  std::vector<BenchmarkReporter::Run> reports;  // return value

  size_t iters = 1;
  std::unique_ptr<internal::ThreadManager> manager;
  std::vector<std::thread> pool(b.threads - 1);
  const int repeats =
      b.repetitions != 0 ? b.repetitions : FLAGS_benchmark_repetitions;
  const bool report_aggregates_only =
      repeats != 1 &&
      (b.report_mode == internal::RM_Unspecified
           ? FLAGS_benchmark_report_aggregates_only
           : b.report_mode == internal::RM_ReportAggregatesOnly);
  for (int i = 0; i < repeats; i++) {
    for (;;) {
      // Try benchmark
      VLOG(2) << "Running " << b.name << " for " << iters << "\n";

      manager.reset(new internal::ThreadManager(b.threads));
      for (std::size_t ti = 0; ti < pool.size(); ++ti) {
        pool[ti] = std::thread(&RunInThread, &b, iters,
                               static_cast<int>(ti + 1), manager.get());
      }
      RunInThread(&b, iters, 0, manager.get());
      manager->WaitForAllThreads();
      for (std::thread& thread : pool) thread.join();
      internal::ThreadManager::Result results;
      {
        MutexLock l(manager->GetBenchmarkMutex());
        results = manager->results;
      }
      manager.reset();
      // Adjust real/manual time stats since they were reported per thread.
      results.real_time_used /= b.threads;
      results.manual_time_used /= b.threads;

      VLOG(2) << "Ran in " << results.cpu_time_used << "/"
              << results.real_time_used << "\n";

      // Base decisions off of real time if requested by this benchmark.
      double seconds = results.cpu_time_used;
      if (b.use_manual_time) {
        seconds = results.manual_time_used;
      } else if (b.use_real_time) {
        seconds = results.real_time_used;
      }

      const double min_time =
          !IsZero(b.min_time) ? b.min_time : FLAGS_benchmark_min_time;
      // If this was the first run, was elapsed time or cpu time large enough?
      // If this is not the first run, go with the current value of iter.
      if ((i > 0) || results.has_error_ || (iters >= kMaxIterations) ||
          (seconds >= min_time) || (results.real_time_used >= 5 * min_time)) {
        BenchmarkReporter::Run report =
            CreateRunReport(b, results, iters, seconds);
        if (!report.error_occurred && b.complexity != oNone)
          complexity_reports->push_back(report);
        reports.push_back(report);
        break;
      }

      // See how much iterations should be increased by
      // Note: Avoid division by zero with max(seconds, 1ns).
      double multiplier = min_time * 1.4 / std::max(seconds, 1e-9);
      // If our last run was at least 10% of FLAGS_benchmark_min_time then we
      // use the multiplier directly. Otherwise we use at most 10 times
      // expansion.
      // NOTE: When the last run was at least 10% of the min time the max
      // expansion should be 14x.
      bool is_significant = (seconds / min_time) > 0.1;
      multiplier = is_significant ? multiplier : std::min(10.0, multiplier);
      if (multiplier <= 1.0) multiplier = 2.0;
      double next_iters = std::max(multiplier * iters, iters + 1.0);
      if (next_iters > kMaxIterations) {
        next_iters = kMaxIterations;
      }
      VLOG(3) << "Next iters: " << next_iters << ", " << multiplier << "\n";
      iters = static_cast<int>(next_iters + 0.5);
    }
  }
  // Calculate additional statistics
  auto stat_reports = ComputeStats(reports);
  if ((b.complexity != oNone) && b.last_benchmark_instance) {
    auto additional_run_stats = ComputeBigO(*complexity_reports);
    stat_reports.insert(stat_reports.end(), additional_run_stats.begin(),
                        additional_run_stats.end());
    complexity_reports->clear();
  }

  if (report_aggregates_only) reports.clear();
  reports.insert(reports.end(), stat_reports.begin(), stat_reports.end());
  return reports;
}

}  // namespace
}  // namespace internal

State::State(size_t max_iters, const std::vector<int>& ranges, int thread_i,
             int n_threads, internal::ThreadTimer* timer,
             internal::ThreadManager* manager)
    : started_(false),
      finished_(false),
      total_iterations_(0),
      range_(ranges),
      bytes_processed_(0),
      items_processed_(0),
      complexity_n_(0),
      error_occurred_(false),
      thread_index(thread_i),
      threads(n_threads),
      max_iterations(max_iters),
      timer_(timer),
      manager_(manager) {
  CHECK(max_iterations != 0) << "At least one iteration must be run";
  CHECK_LT(thread_index, threads) << "thread_index must be less than threads";
}

void State::PauseTiming() {
  // Add in time accumulated so far
  CHECK(started_ && !finished_ && !error_occurred_);
  timer_->StopTimer();
}

void State::ResumeTiming() {
  CHECK(started_ && !finished_ && !error_occurred_);
  timer_->StartTimer();
}

void State::SkipWithError(const char* msg) {
  CHECK(msg);
  error_occurred_ = true;
  {
    MutexLock l(manager_->GetBenchmarkMutex());
    if (manager_->results.has_error_ == false) {
      manager_->results.error_message_ = msg;
      manager_->results.has_error_ = true;
    }
  }
  total_iterations_ = max_iterations;
  if (timer_->running()) timer_->StopTimer();
}

void State::SetIterationTime(double seconds) {
  timer_->SetIterationTime(seconds);
}

void State::SetLabel(const char* label) {
  MutexLock l(manager_->GetBenchmarkMutex());
  manager_->results.report_label_ = label;
}

void State::StartKeepRunning() {
  CHECK(!started_ && !finished_);
  started_ = true;
  manager_->StartStopBarrier();
  if (!error_occurred_) ResumeTiming();
}

void State::FinishKeepRunning() {
  CHECK(started_ && (!finished_ || error_occurred_));
  if (!error_occurred_) {
    PauseTiming();
  }
  // Total iterations now is one greater than max iterations. Fix this.
  total_iterations_ = max_iterations;
  finished_ = true;
  manager_->StartStopBarrier();
}

namespace internal {
namespace {

void RunBenchmarks(const std::vector<Benchmark::Instance>& benchmarks,
                           BenchmarkReporter* console_reporter,
                           BenchmarkReporter* file_reporter) {
  // Note the file_reporter can be null.
  CHECK(console_reporter != nullptr);

  // Determine the width of the name field using a minimum width of 10.
  bool has_repetitions = FLAGS_benchmark_repetitions > 1;
  size_t name_field_width = 10;
  for (const Benchmark::Instance& benchmark : benchmarks) {
    name_field_width =
        std::max<size_t>(name_field_width, benchmark.name.size());
    has_repetitions |= benchmark.repetitions > 1;
  }
  if (has_repetitions) name_field_width += std::strlen("_stddev");

  // Print header here
  BenchmarkReporter::Context context;
  context.num_cpus = NumCPUs();
  context.mhz_per_cpu = CyclesPerSecond() / 1000000.0f;

  context.cpu_scaling_enabled = CpuScalingEnabled();
  context.name_field_width = name_field_width;

  // Keep track of runing times of all instances of current benchmark
  std::vector<BenchmarkReporter::Run> complexity_reports;

  // We flush streams after invoking reporter methods that write to them. This
  // ensures users get timely updates even when streams are not line-buffered.
  auto flushStreams = [](BenchmarkReporter* reporter) {
    if (!reporter) return;
    std::flush(reporter->GetOutputStream());
    std::flush(reporter->GetErrorStream());
  };

  if (console_reporter->ReportContext(context) &&
      (!file_reporter || file_reporter->ReportContext(context))) {
    flushStreams(console_reporter);
    flushStreams(file_reporter);
    for (const auto& benchmark : benchmarks) {
      std::vector<BenchmarkReporter::Run> reports =
          RunBenchmark(benchmark, &complexity_reports);
      console_reporter->ReportRuns(reports);
      if (file_reporter) file_reporter->ReportRuns(reports);
      flushStreams(console_reporter);
      flushStreams(file_reporter);
    }
  }
  console_reporter->Finalize();
  if (file_reporter) file_reporter->Finalize();
  flushStreams(console_reporter);
  flushStreams(file_reporter);
}

std::unique_ptr<BenchmarkReporter> CreateReporter(
    std::string const& name, ConsoleReporter::OutputOptions allow_color) {
  typedef std::unique_ptr<BenchmarkReporter> PtrType;
  if (name == "console") {
    return PtrType(new ConsoleReporter(allow_color));
  } else if (name == "json") {
    return PtrType(new JSONReporter);
  } else if (name == "csv") {
    return PtrType(new CSVReporter);
  } else {
    std::cerr << "Unexpected format: '" << name << "'\n";
    std::exit(1);
  }
}

}  // end namespace
}  // end namespace internal

size_t RunSpecifiedBenchmarks() {
  return RunSpecifiedBenchmarks(nullptr, nullptr);
}

size_t RunSpecifiedBenchmarks(BenchmarkReporter* console_reporter) {
  return RunSpecifiedBenchmarks(console_reporter, nullptr);
}

size_t RunSpecifiedBenchmarks(BenchmarkReporter* console_reporter,
                              BenchmarkReporter* file_reporter) {
  std::string spec = FLAGS_benchmark_filter;
  if (spec.empty() || spec == "all")
    spec = ".";  // Regexp that matches all benchmarks

  // Setup the reporters
  std::ofstream output_file;
  std::unique_ptr<BenchmarkReporter> default_console_reporter;
  std::unique_ptr<BenchmarkReporter> default_file_reporter;
  if (!console_reporter) {
    auto output_opts = ConsoleReporter::OO_None;
    if (FLAGS_benchmark_color == "auto")
      output_opts = IsColorTerminal() ? ConsoleReporter::OO_Color
                                      : ConsoleReporter::OO_None;
    else
      output_opts = IsTruthyFlagValue(FLAGS_benchmark_color)
                        ? ConsoleReporter::OO_Color
                        : ConsoleReporter::OO_None;
    default_console_reporter =
        internal::CreateReporter(FLAGS_benchmark_format, output_opts);
    console_reporter = default_console_reporter.get();
  }
  auto& Out = console_reporter->GetOutputStream();
  auto& Err = console_reporter->GetErrorStream();

  std::string const& fname = FLAGS_benchmark_out;
  if (fname == "" && file_reporter) {
    Err << "A custom file reporter was provided but "
           "--benchmark_out=<file> was not specified."
        << std::endl;
    std::exit(1);
  }
  if (fname != "") {
    output_file.open(fname);
    if (!output_file.is_open()) {
      Err << "invalid file name: '" << fname << std::endl;
      std::exit(1);
    }
    if (!file_reporter) {
      default_file_reporter = internal::CreateReporter(
          FLAGS_benchmark_out_format, ConsoleReporter::OO_None);
      file_reporter = default_file_reporter.get();
    }
    file_reporter->SetOutputStream(&output_file);
    file_reporter->SetErrorStream(&output_file);
  }

  std::vector<internal::Benchmark::Instance> benchmarks;
  if (!FindBenchmarksInternal(spec, &benchmarks, &Err)) return 0;

  if (benchmarks.empty()) {
    Err << "Failed to match any benchmarks against regex: " << spec << "\n";
    return 0;
  }

  if (FLAGS_benchmark_list_tests) {
    for (auto const& benchmark : benchmarks) Out << benchmark.name << "\n";
  } else {
    internal::RunBenchmarks(benchmarks, console_reporter, file_reporter);
  }

  return benchmarks.size();
}

namespace internal {

void PrintUsageAndExit() {
  fprintf(stdout,
          "benchmark"
          " [--benchmark_list_tests={true|false}]\n"
          "          [--benchmark_filter=<regex>]\n"
          "          [--benchmark_min_time=<min_time>]\n"
          "          [--benchmark_repetitions=<num_repetitions>]\n"
          "          [--benchmark_report_aggregates_only={true|false}\n"
          "          [--benchmark_format=<console|json|csv>]\n"
          "          [--benchmark_out=<filename>]\n"
          "          [--benchmark_out_format=<json|console|csv>]\n"
          "          [--benchmark_color={auto|true|false}]\n"
          "          [--v=<verbosity>]\n");
  exit(0);
}

void ParseCommandLineFlags(int* argc, char** argv) {
  using namespace benchmark;
  for (int i = 1; i < *argc; ++i) {
    if (ParseBoolFlag(argv[i], "benchmark_list_tests",
                      &FLAGS_benchmark_list_tests) ||
        ParseStringFlag(argv[i], "benchmark_filter", &FLAGS_benchmark_filter) ||
        ParseDoubleFlag(argv[i], "benchmark_min_time",
                        &FLAGS_benchmark_min_time) ||
        ParseInt32Flag(argv[i], "benchmark_repetitions",
                       &FLAGS_benchmark_repetitions) ||
        ParseBoolFlag(argv[i], "benchmark_report_aggregates_only",
                      &FLAGS_benchmark_report_aggregates_only) ||
        ParseStringFlag(argv[i], "benchmark_format", &FLAGS_benchmark_format) ||
        ParseStringFlag(argv[i], "benchmark_out", &FLAGS_benchmark_out) ||
        ParseStringFlag(argv[i], "benchmark_out_format",
                        &FLAGS_benchmark_out_format) ||
        ParseStringFlag(argv[i], "benchmark_color", &FLAGS_benchmark_color) ||
        // "color_print" is the deprecated name for "benchmark_color".
        // TODO: Remove this.
        ParseStringFlag(argv[i], "color_print", &FLAGS_benchmark_color) ||
        ParseInt32Flag(argv[i], "v", &FLAGS_v)) {
      for (int j = i; j != *argc - 1; ++j) argv[j] = argv[j + 1];

      --(*argc);
      --i;
    } else if (IsFlag(argv[i], "help")) {
      PrintUsageAndExit();
    }
  }
  for (auto const* flag :
       {&FLAGS_benchmark_format, &FLAGS_benchmark_out_format})
    if (*flag != "console" && *flag != "json" && *flag != "csv") {
      PrintUsageAndExit();
    }
  if (FLAGS_benchmark_color.empty()) {
    PrintUsageAndExit();
  }
}

int InitializeStreams() {
  static std::ios_base::Init init;
  return 0;
}

}  // end namespace internal

void Initialize(int* argc, char** argv) {
  internal::ParseCommandLineFlags(argc, argv);
  internal::LogLevel() = FLAGS_v;
}

}  // end namespace benchmark
