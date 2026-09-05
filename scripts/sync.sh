#!/usr/bin/env bash
# Mac <-> 远端 的同步。只有两条通道，方向是单向的，不做双向合并。
#
#   push : Mac scripts/  ->  远端 /root/scripts/            （我写的代码和配置，权威在 Mac）
#   pull : 远端 /workspace/results/  ->  Mac results/       （产物，权威在远端）
#
# RunPod：/workspace 是跨停机保留的 network volume，/root 是停止即清空的 container disk。
# push 的目的地故意还留在 /root/scripts/ —— 权威在 Mac，停机丢了重 push 一次就行，不用占 volume。
#
# 永远不同步的三样：
#   open-unlearning/            —— 两边各自 clone 同一个 commit，§1.7 不许改，靠 commit 保证一致
#   open-unlearning/saves/      —— ckpt，2.5 G 一个，Mac 上没有任何东西需要它
#   envs/ 和 HF 缓存            —— 平台不同，同步没有意义
#
# 用法：
#   HOST=runpod bash scripts/sync.sh push
#   HOST=runpod bash scripts/sync.sh pull
#   HOST=runpod bash scripts/sync.sh push --delete     # 明确要求才删远端多余文件

set -euo pipefail
HOST="${HOST:-runpod}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"; shift || true

# macOS 自带的是 openrsync（protocol 29），不认 --info=progress2 / --partial 这些新参数。
# 有 homebrew 的 rsync 就用它（能开断点续传），没有就退到 openrsync 认识的最小集合。
# `|| true`：两个都不存在时 command -v 返回 1，配上 pipefail 会让 set -e 当场退出
RSYNC="$(command -v /opt/homebrew/bin/rsync /usr/local/bin/rsync 2>/dev/null | head -1 || true)"
if [ -n "$RSYNC" ]; then
    COMMON=(-a -z -h --partial --info=progress2)   # --partial：AutoDL 的 SSH 会断，断点续传
else
    RSYNC=rsync
    COMMON=(-a -z -v)                              # -z：产物几乎全是 JSON，压缩率很高
fi

case "$MODE" in
  push)
    "$RSYNC" "${COMMON[@]}" ${1+"$@"} \
      --exclude='__pycache__/' --exclude='*.pyc' --exclude='.DS_Store' \
      "$LOCAL_ROOT/scripts/" "$HOST:/root/scripts/"
    ;;
  pull)
    mkdir -p "$LOCAL_ROOT/results"
    # 只拉产物。加一道保险：万一有人把 ckpt 写进 results/，别拖回来
    "$RSYNC" "${COMMON[@]}" ${1+"$@"} \
      --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' \
      --exclude='__pycache__/' \
      "$HOST:/workspace/results/" "$LOCAL_ROOT/results/"
    ;;
  *)
    echo "用法: HOST=<sshhost> bash scripts/sync.sh {push|pull} [rsync 额外参数]" >&2
    exit 2
    ;;
esac
