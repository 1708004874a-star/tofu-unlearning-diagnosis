#!/usr/bin/env bash
# 远端环境安装 —— 执行清单_v4.md §1 + §4 段 A 第 1 步
#
# 幂等：可以重复跑，跑到一半断了原样再跑一次即可。
# 目标机：RunPod。Container disk（停止即清空）+ Network volume 挂载在 /workspace（跨停机保留）。
#
# 跟 AutoDL 那版（scripts/remote_setup.autodl.sh，留作存档）的关键区别：
# AutoDL 关机后系统盘还在；RunPod 的 Container disk 官方说明是
# "temporary storage that will be erased when the Pod is stopped"。
# 所以这版把环境、repo、HF 缓存、结果全部放进 /workspace（network volume），
# 不再是"系统盘装环境、数据盘只放 ckpt"那种两处分开的布局——只有一块持久盘，
# 只有真正用完即删、不需要跨停机存活的东西（flash-attn 下载中间文件、
# layer-restore 的临时权重）才留在 /workspace 之外。
#
# 用法：
#   bash remote_setup.sh              # 只装环境，停在数据准备之前
#   RUN_DATA=1 bash remote_setup.sh   # 连 setup_data.py 一起跑（§4 段 A 第 2 步）

set -euo pipefail

REPO_COMMIT="4ad738aaf60f6a4385f6e2506d01da99e76c31f3"   # constants meta.repo_commit
REPO_URL="https://github.com/locuslab/open-unlearning.git"
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/open-unlearning"
ENV_DIR="$WORKSPACE/envs/unlearning"
RESULTS="$WORKSPACE/results"
HF_HOME="$WORKSPACE/.cache/huggingface"
TMP_RESTORE="/root/tmp_restore"          # 用完即删，故意留在 container disk
CONDA_ROOT="$WORKSPACE/miniconda3"       # 装在 volume 上，这样停机也不用重装一遍
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
PYVER="3.11"
RUN_DATA="${RUN_DATA:-0}"

export HF_HOME

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. 机器状态
log "0. 机器状态"
mkdir -p "$RESULTS/env" "$TMP_RESTORE" "$HF_HOME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || die "没有 nvidia-smi"
df -h / "$WORKSPACE"
# network volume 必须真的挂上了，否则下面所有东西都会写进 container disk、一停机就没了
mountpoint -q "$WORKSPACE" || die "$WORKSPACE 不是独立挂载点 —— network volume 没挂上，检查 pod 配置"
# RunPod 的 /workspace 是共享 MooseFS 网络卷，df 的 Avail 是整个机房集群的剩余量（几百 T），
# 不是这个 volume 的配额。以后判断有没有超预算只能看 du 的实际占用，不能看 df 的 Avail。
echo "workspace 实际占用（基线，后面每一步都跟这个比）："
du -sh "$WORKSPACE" 2>/dev/null || true

# ---------------------------------------------------------------- 1. repo
log "1. 抓 open-unlearning @ ${REPO_COMMIT:0:7}"
GIT_TIMEOUT_OPTS=(-c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30)

fetch_repo() {
    rm -rf "$REPO_DIR"
    mkdir -p "$REPO_DIR"
    git -C "$REPO_DIR" init --quiet
    git -C "$REPO_DIR" remote add origin "$REPO_URL"
    # GitHub 允许按 SHA fetch，所以能只抓这一个 commit
    git "${GIT_TIMEOUT_OPTS[@]}" -C "$REPO_DIR" fetch --depth 1 origin "$REPO_COMMIT"
    git -C "$REPO_DIR" checkout --quiet FETCH_HEAD
}

if [ "$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)" != "$REPO_COMMIT" ]; then
    ok=0
    for attempt in 1 2 3; do
        echo "--- 第 $attempt 次 ---"
        if fetch_repo; then ok=1; break; fi
        echo "失败，重试"
        sleep 5
    done
    [ "$ok" = 1 ] || die "三次都抓不下来 —— 改用 Mac 端 rsync 推过来"
fi

ACTUAL=$(git -C "$REPO_DIR" rev-parse HEAD)
[ "$ACTUAL" = "$REPO_COMMIT" ] || die "commit 不对：$ACTUAL"
echo "repo HEAD = $ACTUAL"
# §1.7 不许改库 —— 记指纹，以后随时能证明两端一致且没动过
TREEHASH=$(cd "$REPO_DIR" && git ls-files -s | sha256sum | cut -d' ' -f1)
NFILES=$(cd "$REPO_DIR" && git ls-files | wc -l)
echo "tree hash = $TREEHASH  ($NFILES files)"
[ "$TREEHASH" = "bdb03a6a7ccc811460e48c0790c307d3d3ee527adae4d3269b63ef925c6d202f" ] \
    || die "工作树指纹和 Mac 端对不上"

# ---------------------------------------------------------------- 1b. patches
# §1.7：不改 open-unlearning/ 下任何文件是原则，真改不可时把 patch 存下来、用 git apply 施加。
# 幂等：用 --check --reverse 探测"是不是已经打过"，打过就跳过，不报错。
log "1b. 应用 patches/（§1.7）"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches"
for p in "$PATCH_DIR"/*.patch; do
    [ -f "$p" ] || continue
    name="$(basename "$p")"
    cd "$REPO_DIR"
    if git apply --check --reverse "$p" 2>/dev/null; then
        echo "已打过，跳过：$name"
    elif git apply --check "$p" 2>/dev/null; then
        git apply "$p"
        echo "已应用：$name"
    else
        die "patch 既不能正向应用也不是已应用状态：$name —— 手工检查"
    fi
done

# ---------------------------------------------------------------- 2. saves 目录
log "2. saves 目录"
# AutoDL 那版这里是软链到另一块盘；RunPod 只有一块持久盘，saves 直接建在 repo 里即可
mkdir -p "$REPO_DIR/saves"

# ---------------------------------------------------------------- 3. conda env
log "3. conda env（$ENV_DIR，python $PYVER）"
if [ ! -f "$CONDA_SH" ]; then
    echo "$CONDA_ROOT 下没有 conda，装一个（RunPod 镜像不一定带 conda；装这一次，在 volume 上，以后停机不丢）"
    # 用 Miniforge 不用 Miniconda：base 环境小得多（Miniconda 实测 base 就带 ~4.7G 的包缓存，
    # 是自己装的东西在用不是垃圾，清不掉），而且默认就是 conda-forge 频道，没有下面 ToS 那个坑。
    MINIFORGE_SH="/tmp/miniforge_installer.sh"
    curl -fsSL -o "$MINIFORGE_SH" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$MINIFORGE_SH" -b -p "$CONDA_ROOT"
    rm -f "$MINIFORGE_SH"
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
# defaults 频道（pkgs/main / pkgs/r）现在强制要求先跑一遍交互式 ToS accept 才能用，
# 非交互环境里直接报错退出。这里只是要个干净的 python 解释器，跟 conda-forge 要，绕开这一关。
[ -d "$ENV_DIR" ] || conda create -y -p "$ENV_DIR" --override-channels -c conda-forge "python=$PYVER"
conda activate "$ENV_DIR"
python -VV
[ "$(python -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "$PYVER" ] \
    || die "python 版本不是 $PYVER"

# ---------------------------------------------------------------- 4. 依赖
log "4. pip install -e .（requirements.txt 钉 torch==2.4.1 / transformers==4.51.3）"
cd "$REPO_DIR"
# RunPod 机房在国外，直连官方 PyPI 应该就够快，不像 AutoDL 那版需要换清华源。
# 实测如果还是慢，可以 PIP_INDEX_URL=xxx 覆盖，不影响装的版本（requirements.txt 钉死了）。
if [ -n "${PIP_INDEX_URL:-}" ]; then
    export PIP_INDEX_URL
    echo "pip index: $PIP_INDEX_URL（显式指定）"
else
    echo "pip index: 默认官方源"
fi
pip install --no-cache-dir --upgrade pip setuptools wheel
pip install --no-cache-dir -e .

python -c "import lm_eval" 2>/dev/null || {
    # setup.py 把它声明成 extra（[lm-eval]），但 src/evals/__init__.py 顶层无条件 import 它，
    # 不装的话哪怕只用 TOFU 的指标，src/eval.py 也会在 import 阶段就崩。它只被完全独立的
    # LMEvalEvaluator（MUSE/lm-eval 那条路径）用到，跟 TOFU 指标计算无交集，装它不影响任何数值。
    echo "lm-eval 没装（是 evals/__init__.py 无条件 import 的必需项，不是真正的可选 extra），补装"
    pip install --no-cache-dir "lm-eval==0.4.11"
}

python - <<'PY'
import torch, transformers
print("torch      :", torch.__version__)
print("torch.cuda :", torch.version.cuda)
print("cxx11abi   :", torch._C._GLIBCXX_USE_CXX11_ABI)
print("transformers:", transformers.__version__)
print("device     :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
assert torch.__version__.startswith("2.4.1"), "torch 不是 2.4.1"
assert transformers.__version__ == "4.51.3", "transformers 不是 4.51.3"
PY

# ---------------------------------------------------------------- 5. flash-attn
# §1.1：必装预编译 wheel，不要源码编译。四项必须对上：CUDA 大版本 / torch / python / cxx11abi
log "5. flash-attn 2.6.3（GitHub 预编译 wheel）"
if python -c "import flash_attn" 2>/dev/null; then
    echo "已装：$(python -c 'import flash_attn; print(flash_attn.__version__)')"
else
    NEED_APT=()
    command -v aria2c >/dev/null 2>&1 || NEED_APT+=(aria2)
    command -v unzip  >/dev/null 2>&1 || NEED_APT+=(unzip)
    if [ "${#NEED_APT[@]}" -gt 0 ]; then
        # 不确定 RunPod 镜像的 apt 索引新不新鲜，先 update 一下，免得装不上
        apt-get update -qq && apt-get install -y "${NEED_APT[@]}" >/dev/null
    fi

    read -r TORCH_MM CUDA_TAG ABI <<<"$(python - <<'PY'
import torch
mm  = ".".join(torch.__version__.split("+")[0].split(".")[:2])          # 2.4
cu  = "cu" + torch.version.cuda.split(".")[0] + "3"                     # cu12 -> cu123 标签（小版本互通）
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
print(mm, cu, abi)
PY
)"
    WHEEL="flash_attn-2.6.3+${CUDA_TAG}torch${TORCH_MM}cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl"
    # GitHub release asset URL 里 + 要编码成 %2B，否则代理链路上有些环节会把它当空格处理
    URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/${WHEEL//+/%2B}"
    EXPECT_SIZE=187328293   # GitHub API 查证：这个 wheel 的官方字节数
    echo "wheel: $WHEEL  (期望 $EXPECT_SIZE bytes)"
    OUT="/tmp/${WHEEL}"
    rm -f "$OUT" "${OUT}.aria2"

    # 8 路并发下载，用 kill -0 等 PID 退出（不用 pgrep 模式匹配 —— 在 AutoDL 上实测有过
    # 一次假阴性：进程明明还在，匹配却落空，会让脚本在文件没下完时就往下走）。
    # --file-allocation=none：默认会预分配到目标大小，那样光看文件大小无法判断是否真下完了。
    setsid nohup aria2c -x8 -s8 -k1M --file-allocation=none \
        --retry-wait=3 --max-tries=0 --continue=true \
        --connect-timeout=15 --timeout=30 \
        -d /tmp -o "$WHEEL" "$URL" > /tmp/aria2_flash_attn.log 2>&1 < /dev/null &
    ARIA_PID=$!
    disown
    while kill -0 "$ARIA_PID" 2>/dev/null; do sleep 10; done

    # aria2c 只在真正下完时才删掉 .aria2 控制文件；文件大小相等不能当完成的证据
    # ——在 AutoDL 上曾撞见预分配到目标大小、误读成"已下完"直接拿去装，装出 invalid wheel。
    [ -f "${OUT}.aria2" ] && die "flash-attn 没下完（.aria2 控制文件还在）：$(tail -5 /tmp/aria2_flash_attn.log)"
    [ -f "$OUT" ] || die "flash-attn 下载失败，文件不存在：$(tail -5 /tmp/aria2_flash_attn.log)"
    cur=$(stat -c%s "$OUT")
    [ "$cur" = "$EXPECT_SIZE" ] || die "wheel 大小不对：got $cur, want $EXPECT_SIZE"
    # 大小对不能单独当证据（见上），额外做一次真实内容校验：wheel 是 zip，中央目录读不出来就是坏的
    unzip -tq "$OUT" >/dev/null || die "wheel 大小对但 zip 内容损坏，重新下载"

    pip install --no-cache-dir "$OUT"
    rm -f "$OUT" "${OUT}.aria2"
fi
python -c "import flash_attn, torch; print('flash_attn', flash_attn.__version__, 'import OK')"
# import 通不等于 kernel 能跑，真跑一次前向
python - <<'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16, device="cuda")
o = flash_attn_func(q, q, q)
print("flash_attn kernel OK, out", tuple(o.shape), o.dtype)
PY

# ---------------------------------------------------------------- 6. 只读确认
# §1.3：不是去发现它写的是什么，只为确认 commit 没漂
log "6. 确认 attn_implementation / torch_dtype 没漂"
grep -rn "attn_implementation\|torch_dtype" configs/model/Llama-3.2-1B-Instruct.yaml

# ---------------------------------------------------------------- 7. 数据（默认不跑）
if [ "$RUN_DATA" = "1" ]; then
    log "7. 数据（--eval_logs 和 --idk 都要）"
    python setup_data.py --eval_logs
    python setup_data.py --idk
    ls -la data/idk.jsonl
    REF="saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10/TOFU_EVAL.json"
    [ -f "$REF" ] || die "target(full) 官方日志没下到 —— Gate 1a 要用它"
    [ -f "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json" ] || die "retain90 日志没下到"
    python - <<'PY'
import json
p = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
m = json.load(open(p))["forget_Q_A_ROUGE"]
v = m["value_by_index"]
print("retain90 forget_Q_A_ROUGE agg =", m["agg_value"], "| n =", len(v))
print("per-sample keys:", sorted(next(iter(v.values())).keys()))
assert len(v) == 400
PY
else
    log "7. 数据准备：跳过（RUN_DATA != 1）"
fi

# ---------------------------------------------------------------- 8. 快照 + env.sh
log "8. 环境快照（§6 必留）"
{
  echo "date          : $(date -Iseconds)"
  echo "repo_commit   : $ACTUAL"
  echo "python        : $(python -V 2>&1)"
  echo "torch         : $(python -c 'import torch;print(torch.__version__)')"
  echo "torch.cuda    : $(python -c 'import torch;print(torch.version.cuda)')"
  echo "cxx11abi      : $(python -c 'import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)')"
  echo "transformers  : $(python -c 'import transformers;print(transformers.__version__)')"
  echo "flash_attn    : $(python -c 'import flash_attn;print(flash_attn.__version__)')"
  echo "numpy         : $(python -c 'import numpy;print(numpy.__version__)')"
  echo "accelerate    : $(python -c 'import accelerate;print(accelerate.__version__)')"
  echo "bitsandbytes  : $(python -c 'import bitsandbytes;print(bitsandbytes.__version__)' 2>/dev/null | tail -1)"
  echo "gpu           : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  # constants env.must_record 的三项
  echo "attn_impl     : $(grep -m1 attn_implementation configs/model/Llama-3.2-1B-Instruct.yaml | tr -d ' ')"
  echo "torch_dtype   : $(grep -m1 torch_dtype configs/model/Llama-3.2-1B-Instruct.yaml | tr -d ' ')"
  echo "eval_batch    : $(grep -m1 '^batch_size' configs/eval/tofu.yaml | tr -d ' ')"
  echo "patches       : $(cd "$REPO_DIR" && git diff --stat 2>/dev/null | tail -1 | tr -d ' ' || echo none)"
  for p in "$PATCH_DIR"/*.patch; do [ -f "$p" ] && echo "  - $(basename "$p")"; done
} | tee "$RESULTS/env/env_snapshot.txt"
pip freeze > "$RESULTS/env/pip_freeze.txt"
pip cache purge >/dev/null 2>&1 || true
# conda create 拉下来的包压缩包会留在 miniconda3/pkgs/ 里，装完这一个 env 就再没用了——
# 实测能占到小几个 G，白白吃 volume 配额
conda clean -a -y >/dev/null 2>&1 || true

# 以后每个新脚本、每次新开的 shell，source 这一个文件就能拿到一致的环境——
# 不用重复"激活 conda + 设 HF_HOME + cd 进 repo"三步，也不会漏设 HF_HOME
# 导致模型缓存跑去 container disk、一停机就得重下。
cat > "$WORKSPACE/env.sh" <<EOF
source "$CONDA_SH"
conda activate "$ENV_DIR"
export HF_HOME="$HF_HOME"
cd "$REPO_DIR"
EOF
echo "写了 $WORKSPACE/env.sh —— 以后新 shell: source $WORKSPACE/env.sh"

log "9. 磁盘"
df -h / "$WORKSPACE"
echo "workspace 实际占用（跟配额比，别看上面 df 的 Avail）："
du -sh "$WORKSPACE" 2>/dev/null || true
du -sh "$ENV_DIR" "$REPO_DIR" "$CONDA_ROOT" 2>/dev/null || true

log "完成。快照在 $RESULTS/env/"
