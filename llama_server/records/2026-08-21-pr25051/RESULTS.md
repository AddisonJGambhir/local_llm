> **Historical record - not current operating guidance.**  
> Current instructions: [How to optimize](../../docs/HOW_TO_OPTIMIZE.md) | [Setup](../../docs/SETUP.md)

# PR #25051 -- AMD dual-GPU matrix (2026-08-21)

RX 7900 XTX (Gen4 x16, CPU-direct, ~31.5 GB/s) + AI PRO R9700 (Gen4 x4 via chipset, ~7.9 GB/s).
No P2P; every allreduce round-trips through system memory. Asymmetric 4:1.

Builds: vulkan-pr `13ac90156` (b9834 base) | vulkan-stock `936918514` (b10334) | rocm `030ebb558` (b10358).
Model: Qwen3.8-27B, arch `qwen35`, 65 blocks, HYBRID (carries `qwen35.ssm.*` params).

llama-bench `-r 3 -fa 1 -ub 1024 -b 2048 -p 8192 -n 256`, f16 KV, speculation OFF.
`d0` = KV empty. `d65536` = 65536 real tokens already in KV.

## Q6_K

| backend | split | ratio | depth | pp8192 | tg256 | peak XTX | peak R9700 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rocm | layer | 38/27 | d0 | 1195.16 +/- 0.16 | 24.20 +/- 0.01 | 15699 | 10461 |
| vulkan-pr (#25051) | layer | 28/37 | d0 | 1091.76 +/- 1.19 | 21.78 +/- 0.09 | 11918 | 13807 |
| vulkan-stock | layer | 28/37 | d0 | ABORT (assert) | ABORT (assert) | 11904 | 13462 |
| rocm | row | / | d0 | load failed | load failed | 2358 | 57 |
| vulkan-pr (#25051) | row | / | d0 | 1082.92 +/- 0.57 | 21.69 +/- 0.06 | 11155 | 14314 |
| vulkan-stock | row | / | d0 | load failed | load failed | 2358 | 57 |
| rocm | tensor | 50/50 | d0 | 429.59 +/- 0.44 | 30.27 +/- 0.11 | 14386 | 12047 |
| rocm | tensor | 70/30 | d0 | 386.55 +/- 0.65 | 29.65 +/- 0.13 | 18623 | 7877 |
| vulkan-pr (#25051) | tensor | 50/50 | d0 | 885.56 +/- 5.06 | 23.92 +/- 0.03 | 14365 | 12151 |
| vulkan-pr (#25051) | tensor | 70/30 | d0 | 731.27 +/- 0.75 | 29.77 +/- 0.07 | 18574 | 7949 |
| vulkan-stock | tensor | 50/50 | d0 | ABORT (assert) | ABORT (assert) | 13995 | 11867 |
| vulkan-stock | tensor | 70/30 | d0 | ABORT (assert) | ABORT (assert) | 18157 | 7704 |
| rocm | layer | 38/27 | d65536 | 567.66 +/- 0.14 | 21.31 +/- 0.00 | 18067 | 12763 |
| vulkan-pr (#25051) | layer | 28/37 | d65536 | 461.97 +/- 2.79 | 19.73 +/- 0.05 | 14633 | 17042 |
| vulkan-stock | layer | 28/37 | d65536 | ABORT (assert) | ABORT (assert) | 13757 | 16278 |
| rocm | row | / | d65536 | load failed | load failed | 1907 | 57 |
| vulkan-pr (#25051) | row | / | d65536 | 465.72 +/- 3.76 | 19.44 +/- 0.04 | 14321 | 16423 |
| vulkan-stock | row | / | d65536 | load failed | load failed | 1907 | 57 |
| rocm | tensor | 50/50 | d65536 | 306.63 +/- 0.03 | 27.40 +/- 0.01 | 16437 | 14075 |
| rocm | tensor | 70/30 | d65536 | 275.77 +/- 0.02 | 26.82 +/- 0.03 | 21172 | 9394 |
| vulkan-pr (#25051) | tensor | 50/50 | d65536 | 549.69 +/- 0.58 | 21.79 +/- 0.06 | 16379 | 13980 |
| vulkan-pr (#25051) | tensor | 70/30 | d65536 | 473.33 +/- 0.84 | 26.31 +/- 0.07 | 20932 | 9328 |
| vulkan-stock | tensor | 50/50 | d65536 | ABORT (assert) | ABORT (assert) | 15829 | 13695 |
| vulkan-stock | tensor | 70/30 | d65536 | ABORT (assert) | ABORT (assert) | 20503 | 9020 |

### Q6_K: prefill decay 0 -> 65536

| backend | split | ratio | pp d0 | pp d65536 | change |
| --- | --- | --- | --- | --- | --- |
| rocm | tensor | 50/50 | 429.59 | 306.63 | -28.6% |
| rocm | tensor | 70/30 | 386.55 | 275.77 | -28.7% |
| vulkan-pr (#25051) | tensor | 70/30 | 731.27 | 473.33 | -35.3% |
| vulkan-pr (#25051) | tensor | 50/50 | 885.56 | 549.69 | -37.9% |
| rocm | layer | 38/27 | 1195.16 | 567.66 | -52.5% |
| vulkan-pr (#25051) | row | / | 1082.92 | 465.72 | -57.0% |
| vulkan-pr (#25051) | layer | 28/37 | 1091.76 | 461.97 | -57.7% |

## Q8_0

| backend | split | ratio | depth | pp8192 | tg256 | peak XTX | peak R9700 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rocm | layer | 38/27 | d0 | 1169.85 +/- 0.50 | 21.53 +/- 0.01 | 18520 | 12708 |
| vulkan-pr (#25051) | layer | 28/37 | d0 | 1232.00 +/- 1.29 | 18.74 +/- 0.02 | 14275 | 17034 |
| vulkan-stock | layer | 28/37 | d0 | ABORT (assert) | ABORT (assert) | 14260 | 16603 |
| rocm | row | / | d0 | load failed | load failed | 2358 | 57 |
| vulkan-pr (#25051) | row | / | d0 | 1240.47 +/- 0.32 | 18.89 +/- 0.01 | 13346 | 17707 |
| vulkan-stock | row | / | d0 | load failed | load failed | 2358 | 57 |
| rocm | tensor | 50/50 | d0 | 428.82 +/- 0.41 | 26.54 +/- 0.07 | 17051 | 14768 |
| rocm | tensor | 70/30 | d0 | 383.54 +/- 0.63 | 26.54 +/- 0.06 | 22522 | 9453 |
| vulkan-pr (#25051) | tensor | 50/50 | d0 | 940.78 +/- 0.97 | 21.21 +/- 0.03 | 17192 | 14943 |
| vulkan-pr (#25051) | tensor | 70/30 | d0 | 786.84 +/- 0.58 | 26.01 +/- 0.04 | 22698 | 9582 |
| vulkan-stock | tensor | 50/50 | d0 | ABORT (assert) | ABORT (assert) | 16829 | 14616 |
| vulkan-stock | tensor | 70/30 | d0 | ABORT (assert) | ABORT (assert) | 22146 | 9309 |
| rocm | layer | 38/27 | d65536 | 563.27 +/- 0.05 | 18.97 +/- 0.01 | 20889 | 15009 |
| vulkan-pr (#25051) | layer | 28/37 | d65536 | 468.75 +/- 5.00 | 16.61 +/- 0.01 | 16844 | 20085 |
| vulkan-stock | layer | 28/37 | d65536 | ABORT (assert) | ABORT (assert) | 16113 | 19419 |
| rocm | row | / | d65536 | load failed | load failed | 1907 | 57 |
| vulkan-pr (#25051) | row | / | d65536 | 487.41 +/- 3.75 | 16.75 +/- 0.00 | 15226 | 19730 |
| vulkan-stock | row | / | d65536 | load failed | load failed | 1907 | 57 |
| rocm | tensor | 50/50 | d65536 | 306.95 +/- 0.03 | 24.20 +/- 0.31 | 19891 | 17699 |
| rocm | tensor | 70/30 | d65536 | n/m | n/m | 24333 | 10948 |
| vulkan-pr (#25051) | tensor | 50/50 | d65536 | 574.23 +/- 0.34 | 19.74 +/- 0.06 | 19002 | 16772 |
| vulkan-pr (#25051) | tensor | 70/30 | d65536 | 206.85 +/- 1.12 [GTT-SPILL 23866 MiB, DO NOT CITE] | 11.65 +/- 0.01 [GTT-SPILL 23866 MiB, DO NOT CITE] | 23866 | 10961 |
| vulkan-stock | tensor | 50/50 | d65536 | ABORT (assert) | ABORT (assert) | 18577 | 16444 |
| vulkan-stock | tensor | 70/30 | d65536 | ABORT (assert) | ABORT (assert) | 23387 | 10625 |

### Q8_0: prefill decay 0 -> 65536

| backend | split | ratio | pp d0 | pp d65536 | change |
| --- | --- | --- | --- | --- | --- |
| rocm | tensor | 50/50 | 428.82 | 306.95 | -28.4% |
| vulkan-pr (#25051) | tensor | 50/50 | 940.78 | 574.23 | -39.0% |
| rocm | layer | 38/27 | 1169.85 | 563.27 | -51.9% |
| vulkan-pr (#25051) | row | / | 1240.47 | 487.41 | -60.7% |
| vulkan-pr (#25051) | layer | 28/37 | 1232.00 | 468.75 | -62.0% |

## Q6_K_P

| backend | split | ratio | depth | pp8192 | tg256 | peak XTX | peak R9700 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rocm | layer | 38/27 | d0 | 1206.26 +/- 0.46 | 22.40 +/- 0.00 | 17167 | 11673 |
| vulkan-pr (#25051) | layer | 28/37 | d0 | 1040.16 +/- 4.39 | 20.47 +/- 0.16 | 13079 | 15547 |
| vulkan-stock | layer | 28/37 | d0 | ABORT (assert) | ABORT (assert) | 13064 | 15158 |
| rocm | row | / | d0 | load failed | load failed | 2360 | 57 |
| vulkan-pr (#25051) | row | / | d0 | 1155.51 +/- 0.62 | 20.26 +/- 0.07 | 12233 | 16138 |
| vulkan-stock | row | / | d0 | load failed | load failed | 2360 | 57 |
| rocm | tensor | 50/50 | d0 | 435.28 +/- 0.14 | 27.98 +/- 0.06 | 15949 | 13445 |
| rocm | tensor | 70/30 | d0 | 392.17 +/- 0.13 | 27.48 +/- 0.05 | 20770 | 8664 |
| vulkan-pr (#25051) | tensor | 50/50 | d0 | 903.24 +/- 4.30 | 22.58 +/- 0.07 | 15968 | 13601 |
| vulkan-pr (#25051) | tensor | 70/30 | d0 | 744.26 +/- 0.33 | 28.35 +/- 0.10 | 20950 | 8782 |
| vulkan-stock | tensor | 50/50 | d0 | ABORT (assert) | ABORT (assert) | 15594 | 13294 |
| vulkan-stock | tensor | 70/30 | d0 | ABORT (assert) | ABORT (assert) | 20367 | 8522 |
| rocm | layer | 38/27 | d65536 | 570.23 +/- 0.08 | 19.80 +/- 0.01 | 19531 | 13976 |
| vulkan-pr (#25051) | layer | 28/37 | d65536 | 472.85 +/- 4.95 | 17.74 +/- 0.00 | 14931 | 18363 |
| vulkan-stock | layer | 28/37 | d65536 | ABORT (assert) | ABORT (assert) | 14915 | 17974 |
| rocm | row | / | d65536 | load failed | load failed | 1907 | 57 |
| vulkan-pr (#25051) | row | / | d65536 | 475.66 +/- 3.91 | 17.93 +/- 0.01 | 14069 | 18202 |
| vulkan-stock | row | / | d65536 | load failed | load failed | 1907 | 57 |
| rocm | tensor | 50/50 | d65536 | 307.09 +/- 0.02 | 25.39 +/- 0.26 | 17527 | 15474 |
| rocm | tensor | 70/30 | d65536 | 275.45 +/- 0.04 | 24.56 +/- 0.27 | 22858 | 10184 |
| vulkan-pr (#25051) | tensor | 50/50 | d65536 | 558.91 +/- 0.50 | 20.65 +/- 0.05 | 17375 | 15430 |
| vulkan-pr (#25051) | tensor | 70/30 | d65536 | 481.29 +/- 0.49 | 25.16 +/- 0.11 | 22706 | 10162 |
| vulkan-stock | tensor | 50/50 | d65536 | ABORT (assert) | ABORT (assert) | 16971 | 15122 |
| vulkan-stock | tensor | 70/30 | d65536 | ABORT (assert) | ABORT (assert) | 22256 | 9838 |

### Q6_K_P: prefill decay 0 -> 65536

| backend | split | ratio | pp d0 | pp d65536 | change |
| --- | --- | --- | --- | --- | --- |
| rocm | tensor | 50/50 | 435.28 | 307.09 | -29.5% |
| rocm | tensor | 70/30 | 392.17 | 275.45 | -29.8% |
| vulkan-pr (#25051) | tensor | 70/30 | 744.26 | 481.29 | -35.3% |
| vulkan-pr (#25051) | tensor | 50/50 | 903.24 | 558.91 | -38.1% |
| rocm | layer | 38/27 | 1206.26 | 570.23 | -52.7% |
| vulkan-pr (#25051) | layer | 28/37 | 1040.16 | 472.85 | -54.5% |
| vulkan-pr (#25051) | row | / | 1155.51 | 475.66 | -58.8% |

Runs parsed: 72
