# 0017. Unset thread counts are one per physical core the process may use

Date: 2026-11-19

## Context

A `decode_threads` left `None` let llama.cpp decide, which is 4 threads on any
machine, and `prefill_threads` took as many. llama.cpp's workers spin while
they wait, so a thread beyond what the processor can give is throttled rather
than idle.

## Decision

- `decode_threads` left `None` is one thread per physical core of the
  processors the process may run on. `prefill_threads` left `None` is as many
  as `decode_threads`. A count the caller sets wins.
- The processors are the process's affinity where the system reports one,
  else all of them. Physical cores come from each processor's package and core
  under `/sys/devices/system/cpu`. Where the system lists performance cores
  apart, only those are counted.
- Where the topology cannot be read (macOS, Windows, a container without
  `/sys`), the count is llama.cpp's own fallback: every thread up to four, half
  of them beyond.
- A Linux CPU quota bounds the count: the groups the process is in are read
  from `/proc/self/cgroup`, every group from each up to the root is read
  (cgroup v2 `cpu.max`, v1 `cpu.cfs_quota_us` with `cpu.cfs_period_us`), and
  the tightest quota, rounded down and at least one, bounds the cores. Where
  `/proc/self/cgroup` cannot be read, the top group alone is read.
- The counts are read once at load, with the standard library alone.

## Consequences

Measured on an i9-9900K (8 cores, 16 threads, processor only, context 4096,
batch 512, a history of about 1,040 tokens, 64 generated tokens, greedy),
interleaved runs while other processes raised the load average from 1 to 14:

| Model | Threads (decode, prefill) | Prefill, tokens/s | Decode, tokens/s |
|---|---|---|---|
| gemma-3-1b Q4_K_M | 4, 4 | 173.8, 172.4, 173.4 | 37.8, 37.0, 38.6 |
| gemma-3-1b Q4_K_M | 8, 16 | 335.6, 56.0, 51.8 | 45.1, 44.0, 43.5 |
| gemma-3-1b Q4_K_M | 8, 8 | 327.1, 209.3, 213.8 | 45.5, 41.5, 43.4 |
| Llama-3.2-3B Q4_K_M | 4, 4 | 73.9, 72.9, 73.4 | 14.0, 13.8, 13.8 |
| Llama-3.2-3B Q4_K_M | 8, 16 | 134.8, 49.8, 51.1 | 17.0, 16.8, 16.0 |
| Llama-3.2-3B Q4_K_M | 8, 8 | 105.0, 104.0, 133.1 | 16.4, 16.9, 16.7 |

Decode on the physical cores is 1.15 to 1.2 times as fast as on 4 threads
under every load. Prefill on the physical cores stays at 1.2 to 1.9 times the
speed of 4 threads under every load.

## Alternatives rejected

- llama.cpp's default of 4 threads: slower on every machine with more cores.
- Prefill on every thread: about as fast as the physical cores on an idle
  machine, but 50 to 70 tokens/s in 9 of 10 runs on a busy one, under half
  the old 4 threads, and the load's first decodes, which run on the prefill
  threads, took up to 8.6 s instead of 1.0 s.
- Reading only the top of `/sys/fs/cgroup`: on a host that is the root group,
  so a quota on the service the process runs in was missed.
