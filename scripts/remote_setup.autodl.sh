#!/usr/bin/env bash
# 远端环境安装 —— 执行清单_v4.md §1 + §4 段 A 第 1 步
#
# 幂等：可以重复跑，跑到一半断了原样再跑一次即可。
# 目标机：AutoDL，系统盘 30 G（/）+ 数据盘 50 G（/root/autodl-tmp）。
# 原则：环境、repo、HF 缓存全在系统盘；数据盘只放 ckpt。
#
# 网络（AutoDL 特有）：
#   network_turbo 只对 github / huggingface 有用，开着会让 pip 源*更慢*（官方原话）。
#   所以代理只在 git clone 和 GitHub wheel 下载那两处的子 shell 里开，pip 一律不开。
#
# 用法：
#   bash remote_setup.sh              # 只装环境，停在数据准备之前
#   RUN_DATA=1 bash remote_setup.sh   # 连 setup_data.py 一起跑（§4 段 A 第 2 步）

set -euo pipefail

REPO_COMMIT="4ad738aaf60f6a4385f6e2506d01da99e76c31f3"   # constants meta.repo_commit
REPO_URL="https://github.com/locuslab/open-unlearning.git"
REPO_DIR="/root/open-unlearning"
ENV_DIR="/root/envs/unlearning"
DATA_SAVES="/root/autodl-tmp/saves"
RESULTS="/root/results"
TMP_RESTORE="/root/tmp_restore"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
PYVER="3.11"
RUN_DATA="${RUN_DATA:-0}"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }
# 代理只在这个函数里生效，函数返回后环境变量随子 shell 一起消失
with_turbo() { ( source /etc/network_turbo >/dev/null 2>&1; "$@" ); }

# ---------------------------------------------------------------- 0. 机器状态
log "0. 机器状态"
mkdir -p "$RESULTS/env" "$TMP_RESTORE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || die "没有 nvidia-smi"
df -h / /root/autodl-tmp
# 数据盘必须存在，否则软链会把 ckpt 写到系统盘、30 G 当场爆
mountpoint -q /root/autodl-tmp || die "/root/autodl-tmp 不是独立挂载点 —— 方案不成立"

# ---------------------------------------------------------------- 1. repo
# AutoDL 的 network_turbo 代理会断流：git 不会报错，就那么挂着（实测卡在 6.8 M）。
# 对策三条：① 只抓需要的那一个 commit（--depth 1，几 M 而不是几十 M）；
#          ② lowSpeedLimit/lowSpeedTime 让停流的连接直接失败而不是永远等；③ 重试 3 次。
log "1. 抓 open-unlearning @ ${REPO_COMMIT:0:7}（走 network_turbo）"
GIT_TIMEOUT_OPTS=(-c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30)

fetch_repo() {
    rm -rf "$REPO_DIR"
    mkdir -p "$REPO_DIR"
    git -C "$REPO_DIR" init --quiet
    git -C "$REPO_DIR" remote add origin "$REPO_URL"
    # GitHub 允许按 SHA fetch，所以能只抓这一个 commit
    with_turbo git "${GIT_TIMEOUT_OPTS[@]}" -C "$REPO_DIR" fetch --depth 1 origin "$REPO_COMMIT"
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
# ---------------------------------------------------------------- 2. saves 软链
log "2. saves 软链 -> 数据盘"
mkdir -p "$DATA_SAVES"
if [ -e "$REPO_DIR/saves" ] && [ ! -L "$REPO_DIR/saves" ]; then
    die "$REPO_DIR/saves 已存在且不是软链 —— 手工处理，不要盲删"
fi
ln -sfn "$DATA_SAVES" "$REPO_DIR/saves"
echo "saves -> $(readlink -f "$REPO_DIR/saves")"

# ---------------------------------------------------------------- 3. conda env
log "3. conda env（系统盘，python $PYVER）"
[ -f "$CONDA_SH" ] || die "找不到 $CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH"
[ -d "$ENV_DIR" ] || conda create -y -p "$ENV_DIR" "python=$PYVER"
conda activate "$ENV_DIR"
python -VV
[ "$(python -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "$PYVER" ] \
    || die "python 版本不是 $PYVER"

# ---------------------------------------------------------------- 4. 依赖
# --no-cache-dir：装 torch 之后 ~/.cache/pip 能到 3–5 G，全压在系统盘上
# 不开 network_turbo：pip 走 aliyun 镜像，开代理反而更慢
log "4. pip install -e .（requirements.txt 钉 torch==2.4.1 / transformers==4.51.3）"
cd "$REPO_DIR"
# 机器自带的 /etc/pip.conf 指向 aliyun，实测 0.12 MB/s（torch 2.5 G 要跑 6 小时）；
# 清华同一个文件 3.07 MB/s。换源不改装什么版本（requirements.txt 钉死了），不是偏离。
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
echo "pip index: $PIP_INDEX_URL"
pip install --no-cache-dir --upgrade pip setuptools wheel
pip install --no-cache-dir -e .

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
log "5. flash-attn 2.6.3（GitHub 预编译 wheel，走 network_turbo）"
if python -c "import flash_attn" 2>/dev/null; then
    echo "已装：$(python -c 'import flash_attn; print(flash_attn.__version__)')"
else
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

    # 实测单连接 curl 在这条代理链路上会被限速到 10–30 KB/s 且持续走低（187M 要几小时），
    # 还会在中途被掐断（曾在 63M 处 curl:(18)）。8 路并发的 aria2c 实测 ~1.2 MB/s，快一个数量级。
    # 用 kill -0 等 PID 退出，而不是 pgrep 模式匹配——实测 pgrep 轮询有过一次假阴性（进程明明还在，
    # 匹配却落空），会让脚本在文件没下完时就往下走。
    # --file-allocation=none：默认会预分配到目标大小，那样光看文件大小无法判断是否真下完了。
    command -v aria2c >/dev/null 2>&1 || apt-get install -y aria2 >/dev/null
    with_turbo setsid nohup aria2c -x8 -s8 -k1M --file-allocation=none \
        --retry-wait=3 --max-tries=0 --continue=true \
        --connect-timeout=15 --timeout=30 \
        -d /tmp -o "$WHEEL" "$URL" > /tmp/aria2_flash_attn.log 2>&1 < /dev/null &
    ARIA_PID=$!
    disown
    while kill -0 "$ARIA_PID" 2>/dev/null; do sleep 10; done

    # aria2c 只在真正下完时才删掉 .aria2 控制文件；文件大小相等不能当完成的证据
    # ——曾撞见 aria2c 预分配到目标大小、脚本另一次运行误读成"已下完"直接拿去装，装出 invalid wheel。
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
    log "7. 数据（--eval_logs 和 --idk 都要，走 network_turbo）"
    with_turbo python setup_data.py --eval_logs
    with_turbo python setup_data.py --idk
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

# ---------------------------------------------------------------- 8. 快照
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
} | tee "$RESULTS/env/env_snapshot.txt"
pip freeze > "$RESULTS/env/pip_freeze.txt"
pip cache purge >/dev/null 2>&1 || true

log "9. 磁盘"
df -h / /root/autodl-tmp
du -sh "$ENV_DIR" "$REPO_DIR" 2>/dev/null || true

log "完成。快照在 $RESULTS/env/"
