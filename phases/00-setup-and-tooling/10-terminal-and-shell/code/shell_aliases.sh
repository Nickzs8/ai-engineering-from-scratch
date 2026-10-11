#!/usr/bin/env bash
#
# Aliases e funções para desenvolvimento de IA com GPU AMD (ROCm).
# Adaptado de shell_aliases.sh (lição 10, Terminal & Shell), que usa NVIDIA.
#
# Use no ~/.bashrc:
#   source /caminho/para/shell_aliases_amd.sh

# --- GPU (AMD: rocm-smi no lugar de nvidia-smi) ---
# rocm-smi vem com o ROCm instalado no sistema.
# Se não tiver, "nvtop" (sudo apt install nvtop) também lê GPUs AMD.

alias gpu='rocm-smi --showproductname --showuse --showmemuse --showtemp'
alias gpuwatch='watch -n1 rocm-smi'
alias gpumem='rocm-smi --showmeminfo vram'
alias gpuprocs='rocm-smi --showpids'

# --- Controle de treino ---

alias killtraining='pkill -f "python.*train"'

killtrain() {
    if [ -z "$1" ]; then
        pkill -f "python.*train"
        echo "Todos os processos python de treino foram encerrados"
    else
        pkill -f "$1"
        echo "Processos encerrados: $1"
    fi
}

# --- Ambientes virtuais ---

alias ae='source .venv/bin/activate'
alias de='deactivate'
alias mkvenv='python3 -m venv .venv && source .venv/bin/activate'
alias uvvenv='uv venv && source .venv/bin/activate'

# --- Acompanhar logs ---

alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
alias watchacc='tail -f logs/*.log | grep --line-buffered "accuracy\|acc"'
alias watcherr='tail -f logs/*.log | grep --line-buffered "ERROR\|error\|Exception"'

taillog() {
    local pattern="${1:-loss}"
    tail -f logs/*.log 2>/dev/null | grep --line-buffered "$pattern"
}

# --- Espaço em disco ---

alias diskuse='df -h .'
alias bigfiles='find . -type f -size +100M | xargs du -h 2>/dev/null | sort -rh | head -20'
alias bigmodels='find . \( -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" -o -name "*.ckpt" -o -name "*.bin" \) | xargs du -h 2>/dev/null | sort -rh | head -20'

# --- Verificações rápidas do ambiente ---
# No PyTorch para ROCm, torch.cuda.is_available() também funciona com AMD,
# e torch.version.hip mostra a versão do HIP (None se o torch não for ROCm).

checkgpu() {
    python3 - <<'EOF'
import torch
print(f"GPU disponível: {torch.cuda.is_available()}")
print(f"HIP: {torch.version.hip}")
if torch.cuda.is_available():
    print(f"Dispositivo: {torch.cuda.get_device_name(0)}")
EOF
}

alias checkrocm='env | grep -i -E "rocm|hip|hsa"'

checkenv() {
    python3 --version
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, HIP {torch.version.hip}, GPU {torch.cuda.is_available()}')" 2>/dev/null \
        || echo "PyTorch não encontrado neste ambiente"
}

# --- Atalhos do tmux ---

alias ta='tmux attach -t'
alias tls='tmux ls'
alias tn='tmux new -s'
alias tk='tmux kill-session -t'

trainenv() {
    local name="${1:-train}"
    tmux new-session -d -s "$name"
    tmux split-window -h -t "$name"
    tmux split-window -v -t "$name"
    tmux send-keys -t "$name:0.1" 'watch -n1 rocm-smi' C-m
    tmux send-keys -t "$name:0.2" 'htop' C-m
    tmux select-pane -t "$name:0.0"
    tmux attach -t "$name"
}

# --- SSH ---

syncto() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Uso: syncto <host> <caminho_remoto> [caminho_local]"
        echo "Exemplo: syncto gpu ~/data ./data"
        return 1
    fi
    local host="$1"
    local remote="$2"
    local local_path="${3:-.}"
    rsync -avz --progress "$local_path" "${host}:${remote}"
}

syncfrom() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Uso: syncfrom <host> <caminho_remoto> [caminho_local]"
        echo "Exemplo: syncfrom gpu ~/results ./results"
        return 1
    fi
    local host="$1"
    local remote="$2"
    local local_path="${3:-.}"
    rsync -avz --progress "${host}:${remote}" "$local_path"
}

# --- Gerenciamento de experimentos ---

newexp() {
    local name="${1:-experiment}"
    local dir="experiments/${name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$dir/logs" "$dir/checkpoints" "$dir/configs"
    echo "Pasta do experimento criada: $dir"
    echo "$dir"
}

lastexp() {
    ls -dt experiments/*/ 2>/dev/null | head -1
}

# --- Download de modelos ---

hfdownload() {
    if [ -z "$1" ]; then
        echo "Uso: hfdownload <model_id> [arquivo]"
        echo "Exemplo: hfdownload meta-llama/Llama-2-7b config.json"
        return 1
    fi
    local model="$1"
    local file="${2:-}"
    if [ -n "$file" ]; then
        wget "https://huggingface.co/${model}/resolve/main/${file}"
    else
        echo "Clonando o repositório inteiro (precisa do git-lfs)..."
        git lfs install
        git clone "https://huggingface.co/${model}"
    fi
}

# --- Gerenciamento de processos ---

memhogs() {
    ps aux --sort=-%mem 2>/dev/null | head -11 || ps aux -m | head -11
}

psg() {
    ps aux | grep -v grep | grep -i "$1"
}