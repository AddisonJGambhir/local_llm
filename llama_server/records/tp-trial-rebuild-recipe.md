# tp-trial rebuild recipe (2026-08-22)

What this build is:
  PR #25051 (Vulkan tensor allreduce, 14 commits) rebased onto llama.cpp master
  b21e4de74 (2026-08-22) + PR #27210 (adaptive MTP) + the 12-line vk_queue port
  below. Produces the A1+A7 trial binary (F21 in DECODE_SPEED_REPORT.md).

Live locations (durable):
  Source worktree: ~/Desktop/local_llm/llama.cpp-tp-trial
  Build dir:       ~/Desktop/local_llm/llama.cpp-tp-trial/build  (in-worktree, gitignored by /llama.cpp*/)
  Binaries:        ~/Desktop/local_llm/llama.cpp-tp-trial/build/bin
  RUNPATH is absolute to that bin dir: the binaries run from any cwd with no
  LD_LIBRARY_PATH. Moving or renaming the build dir BREAKS them (they die with
  'libllama-server-impl.so: cannot open shared object file') -- rebuild instead of mv.

Rebuild from scratch if the worktree is ever lost:
  cd ~/Desktop/local_llm/llama.cpp
  git fetch origin master
  git fetch origin pull/25051/head:pr25051      # or use an existing local pr25051
  MB=$(git merge-base pr25051 origin/master)    # 3fc4e1052 at creation time
  git worktree add --detach ../llama.cpp-tp-trial pr25051
  cd ../llama.cpp-tp-trial
  git rebase --onto origin/master $MB HEAD      # applied with ZERO conflicts 2026-08-22
  git apply ~/Desktop/local_llm/llama_server/records/vk-queue-unique-ptr-port.patch
  git apply ~/Desktop/local_llm/llama_server/records/pr27210-adaptive-mtp.diff
  cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_CCACHE=ON
  cmake --build build -j20      # 841 targets, ~6 min on this box with ccache warm

Port patch note (records/vk-queue-unique-ptr-port.patch):
  Master converted vk_device queues to std::unique_ptr<vk_queue>; the PR's
  comm-init loop still used raw pointers. The patch null-safes the transfer
  queue (single-queue devices have transfer_queue == nullptr) and uses
  ->handle->queue for the raw VkQueue comparison.
