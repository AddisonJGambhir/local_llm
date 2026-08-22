# tp-trial rebuild recipe (2026-08-22)

What this build is:
  PR #25051 (Vulkan tensor allreduce, 14 commits) rebased onto llama.cpp master
  b21e4de74 (2026-08-22) + PR #27210 (adaptive MTP) + the 12-line vk_queue port
  below. Produces the A1+A7 trial binary (F21 in DECODE_SPEED_REPORT.md).

Live locations (durable):
  Source worktree: ~/Desktop/local_llm/llama.cpp-tp-trial
  Build dir:       ~/tmp-tp-trial/build   (binaries: ~/tmp-tp-trial/build/bin)

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
  cmake -G Ninja -S . -B ~/tmp-tp-trial/build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_CUDA=OFF
  cmake --build ~/tmp-tp-trial/build -j16       # ~10 min on this box

Port patch note (records/vk-queue-unique-ptr-port.patch):
  Master converted vk_device queues to std::unique_ptr<vk_queue>; the PR's
  comm-init loop still used raw pointers. The patch null-safes the transfer
  queue (single-queue devices have transfer_queue == nullptr) and uses
  ->handle->queue for the raw VkQueue comparison.
