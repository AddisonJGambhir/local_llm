# Backlog: things that need sudo or a reboot

Everything here is blocked on root or a restart, so an agent cannot do it.
Each entry says what to run, what it should buy, and how to verify or undo it.

Last updated 2026-08-21.

---

## 1. PCIe ASPM -> performance mode   [sudo, no reboot, reverts on reboot]

**Expected gain: +10.8% dense decode.** Highest-value item here.

Reported in llama.cpp discussion #21043 (RDNA4/R9700 tuning): 29.30 -> 32.46 t/s
on a dense model under RADV. Minimal effect on MoE (+0.5%).

Current state on this box: `[default]` (verified 2026-08-21, NOT performance).

```bash
cat /sys/module/pcie_aspm/parameters/policy      # shows [default]
sudo sh -c 'echo performance > /sys/module/pcie_aspm/parameters/policy'
cat /sys/module/pcie_aspm/parameters/policy      # should show [performance]
```

Undo: reboot, or echo `default` back.

WHY IT MAY MATTER MORE HERE THAN IN THE LINKED DISCUSSION: the R9700 sits behind
five chipset bridges on a Gen4 x4 link. ASPM link power-state transitions add
latency on exactly that path, and tensor-split decode crosses it constantly.

To verify after applying, re-run the current champion and compare against 33.21:
```bash
cd ~/Desktop/local_llm
env -u LD_LIBRARY_PATH GGML_CUDA_DISABLE_GRAPHS=1 \
  ./llama.cpp/build/bin/llama-bench -m llama_server/models/Qwen3.8-27B-Q6_K.gguf \
  -sm tensor -dev ROCm0/ROCm1 -ts 58/42 -mg 1 -ngl 999 -fa 1 \
  -ub 256 -b 2048 -p 0 -n 512 -r 5
```

---

## 2. `iommu=pt` kernel parameter   [sudo + reboot]

**Expected gain: probably none. Diagnostic only.** Listed for completeness.

The IOMMU is in translating mode (38 groups, `iommu=pt` absent from
/proc/cmdline). AMD's own troubleshooting names this as a cause of multi-GPU
RCCL instability.

```bash
sudo nano /etc/default/grub
# add iommu=pt to GRUB_CMDLINE_LINUX_DEFAULT
sudo update-grub && sudo reboot
cat /proc/cmdline    # confirm iommu=pt present
```

WHY I EXPECT IT TO FAIL: RCCL here crashes on the host SHM transport too, not
just direct P2P (measured with NCCL_P2P_DISABLE=1). `iommu=pt` addresses DMA
remapping on the P2P path, which is not the path that dies. See
docs/pr25051-run-2026-08-21/RCCL_FINDINGS.md.

Even if it fixed the crash, there is no P2P route between these cards, so RCCL
would fall back to host memory -- the same transport llama.cpp's internal
allreduce already uses, which already measures identically to the alternative
(430.30 vs 429.57 pp).

---

## 3. Replace ROCm with AMD's own 7.14 stack   [sudo, large, reboot]

**Expected gain: fixes the RCCL crash. Probably no speed change.**

This box runs Ubuntu's ROCm packaging (`7.1.0-0ubuntu2`) plus the in-tree
mainline amdgpu driver. AMD's support matrix does not cover Ubuntu 26.04 until
ROCm 7.14. RCCL 2.72.5 as packaged cannot complete a two-GPU AllReduce -- AMD's
own `rccl-UnitTests` dies on the first collective with "Child 0 pipe closed
unexpectedly", with llama.cpp entirely out of the picture.

Involves adding repo.radeon.com, installing amdgpu-dkms (out-of-tree driver),
and replacing the ROCm userspace. Do NOT mix individual 7.1 and 7.14 packages.

Validate BEFORE rebuilding llama.cpp:
```bash
UT_MIN_GPUS=2 /usr/libexec/rocm/librccl1-tests/rccl-UnitTests --gtest_filter='AllReduce.*'
```
Only if that passes is it worth rebuilding with `-DGGML_HIP_RCCL=ON`. The
`llama.cpp/build-rccl/` tree is already configured; it is a ~90 second rebuild.

RISK: replaces a working GPU stack on a machine whose desktop runs off the XTX.

---

## 4. BIOS: confirm PCIe link config   [reboot into firmware]

**Expected gain: probably none. Verification only.**

CORRECTED 2026-08-21: an earlier version of this file claimed the R9700 could be
moved to a CPU-attached slot via BIOS. That is WRONG and has been removed. Lane
routing is fixed by board wiring and which physical slot the card occupies; no
firmware setting reroutes it.

Measured routing:
- 7900 XTX  `03:00.0` <- `01:00.0` <- root port `00:01.1`, Gen4 x16 CPU-direct.
  16.0 GT/s is CORRECT and not a misconfiguration: the 7900 XTX is a PCIe 4.0
  card and cannot negotiate Gen5.
- R9700 `14:00.0` <- `13:00.0` <- `12:00.0` <- four AMD 600-series CHIPSET
  bridges (`0c:08.0`, `0b:00.0`, `06:08.0`, `05:00.0`) <- root port `00:02.1`,
  which is a CPU root port but wired x4. Effective 16.0 GT/s x4, ~7.9 GB/s.

The only CPU-attached x16 root ports are `00:01.1` (the XTX) and `00:08.1` /
`00:08.3`, which are internal devices rather than usable slots. So even a
physical move probably has nowhere better to go on this board.

Still worth an eyeball in firmware, but expect nothing:
Above 4G Decoding, Resizable BAR, and any PCIe link-speed caps.

## Not blocked, for contrast

These need no sudo and are already done or doable by an agent:
- `GGML_CUDA_DISABLE_GRAPHS=1` -- +1.3% to +6.0% decode, already measured
- tensor-split ratio tuning -- 58/42 ROCm, 75/25 Vulkan, already measured
- llama.cpp rebuilds into separate build dirs -- no root needed
