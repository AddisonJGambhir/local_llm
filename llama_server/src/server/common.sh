#!/usr/bin/env bash

LLAMA_SERVER_ROOT="${LLAMA_SERVER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LLM_ROOT="${LLM_ROOT:-$(cd "$LLAMA_SERVER_ROOT/.." && pwd)}"

llama_die() {
    echo "error: $*" >&2
    return 1
}

llama_physical_cores() {
    local count
    count="$(
        lscpu -p=CORE,SOCKET 2>/dev/null |
            awk -F, '!/^#/ { seen[$1 "," $2] = 1 } END { print length(seen) }'
    )"
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
        count="$(nproc)"
    fi
    printf '%s\n' "$count"
}

llama_load_profile() {
    local profile_path="$1"

    [[ -r "$profile_path" ]] || llama_die "profile is not readable: $profile_path" || return

    # Profiles are trusted shell configuration owned by this setup.
    # shellcheck source=/dev/null
    source "$profile_path"

    : "${PROFILE_NAME:=unnamed}"
    : "${LLAMA_ALIAS:=local}"
    : "${LLAMA_BACKEND:=vulkan}"
    : "${LLAMA_DEVICE:=Vulkan0}"
    : "${LLAMA_GPU_LAYERS:=99}"
    : "${LLAMA_CONTEXT:=131072}"
    : "${LLAMA_THREADS:=auto}"
    : "${LLAMA_BATCH_THREADS:=auto}"
    : "${LLAMA_PARALLEL:=1}"
    : "${LLAMA_UBATCH:=192}"
    : "${LLAMA_CACHE_TYPE_K:=q8_0}"
    : "${LLAMA_CACHE_TYPE_V:=q8_0}"
    : "${LLAMA_FLASH_ATTN:=on}"
    : "${LLAMA_CONTEXT_SHIFT:=0}"
    : "${LLAMA_SPEC_MODE:=none}"
    : "${LLAMA_TEMPERATURE:=0.6}"
    : "${LLAMA_TOP_P:=0.95}"
    : "${LLAMA_TOP_K:=20}"
    : "${LLAMA_MIN_P:=0.0}"
    : "${LLAMA_PRESENCE_PENALTY:=0.0}"
    : "${LLAMA_REPEAT_PENALTY:=1.0}"
    : "${LLAMA_REASONING:=auto}"
    : "${LLAMA_CHAT_TEMPLATE_KWARGS:=}"
    : "${LLAMA_HOST:=127.0.0.1}"
    : "${LLAMA_PORT:=1234}"
    : "${LLAMA_API_KEY_FILE:=}"
    : "${LLAMA_TIMEOUT:=0}"
    : "${LLAMA_JINJA:=1}"
    : "${LLAMA_METRICS:=0}"
    : "${LLAMA_STARTUP_TIMEOUT:=300}"
    : "${LLAMA_STOP_TIMEOUT:=30}"
    : "${LLAMA_LOG_MAX_BYTES:=52428800}"
    : "${LLAMA_LOG_BACKUPS:=5}"
    : "${LLAMA_SYNC_CLIENTS:=1}"

    if [[ "$LLAMA_THREADS" == "auto" ]]; then
        LLAMA_THREADS="$(llama_physical_cores)"
    fi
    if [[ "$LLAMA_BATCH_THREADS" == "auto" ]]; then
        LLAMA_BATCH_THREADS="$(nproc)"
    fi

    LLAMA_STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}/local-llm"
    LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-$LLAMA_STATE_HOME/llama-server.log}"

    if ! declare -p LLAMA_EXTRA_ARGS &>/dev/null; then
        LLAMA_EXTRA_ARGS=()
    fi
}

llama_validate_profile() {
    local host_is_loopback=0

    [[ -n "${LLAMA_BINARY:-}" ]] || llama_die "LLAMA_BINARY is not set" || return
    [[ -x "$LLAMA_BINARY" ]] || llama_die "server binary is not executable: $LLAMA_BINARY" || return
    [[ -n "${LLAMA_MODEL:-}" ]] || llama_die "LLAMA_MODEL is not set" || return
    [[ -r "$LLAMA_MODEL" ]] || llama_die "model is not readable: $LLAMA_MODEL" || return
    [[ "$LLAMA_PORT" =~ ^[0-9]+$ ]] && ((LLAMA_PORT >= 1 && LLAMA_PORT <= 65535)) ||
        llama_die "invalid port: $LLAMA_PORT" || return
    [[ "$LLAMA_CONTEXT" =~ ^[1-9][0-9]*$ ]] || llama_die "invalid context: $LLAMA_CONTEXT" || return
    [[ "$LLAMA_THREADS" =~ ^[1-9][0-9]*$ ]] || llama_die "invalid thread count: $LLAMA_THREADS" || return
    [[ "$LLAMA_BATCH_THREADS" =~ ^[1-9][0-9]*$ ]] ||
        llama_die "invalid batch thread count: $LLAMA_BATCH_THREADS" || return

    case "$LLAMA_HOST" in
        127.0.0.1|localhost|::1) host_is_loopback=1 ;;
    esac
    if ((host_is_loopback == 0)) && [[ -z "$LLAMA_API_KEY_FILE" ]]; then
        llama_die "refusing non-loopback host '$LLAMA_HOST' without LLAMA_API_KEY_FILE" || return
    fi
    if [[ -n "$LLAMA_API_KEY_FILE" && ! -r "$LLAMA_API_KEY_FILE" ]]; then
        llama_die "API-key file is not readable: $LLAMA_API_KEY_FILE" || return
    fi

    case "$LLAMA_BACKEND" in
        vulkan|rocm) ;;
        *) llama_die "unsupported backend: $LLAMA_BACKEND" || return ;;
    esac
    case "$LLAMA_SPEC_MODE" in
        none|mtp|dflash) ;;
        *) llama_die "unsupported speculative mode: $LLAMA_SPEC_MODE" || return ;;
    esac
    if [[ "$LLAMA_SPEC_MODE" == "dflash" && ! -r "${LLAMA_DRAFTER:-}" ]]; then
        llama_die "DFlash drafter is not readable: ${LLAMA_DRAFTER:-<unset>}" || return
    fi
    if [[ -n "$LLAMA_CHAT_TEMPLATE_KWARGS" ]]; then
        python3 -c 'import json,sys; json.loads(sys.argv[1])' "$LLAMA_CHAT_TEMPLATE_KWARGS" ||
            llama_die "LLAMA_CHAT_TEMPLATE_KWARGS is not valid JSON" || return
    fi
}

llama_build_command() {
    LLAMA_CMD=(
        "$LLAMA_BINARY"
        -m "$LLAMA_MODEL"
        --alias "$LLAMA_ALIAS"
    )

    if [[ "$LLAMA_BACKEND" == "vulkan" ]]; then
        LLAMA_CMD+=(--device "$LLAMA_DEVICE")
    fi

    LLAMA_CMD+=(
        -ngl "$LLAMA_GPU_LAYERS"
        # BeeLlama v0.4.x makes a failed auto-fit fatal, and auto-fit always aborts
        # when -ngl is set explicitly (it always is here). Context is hand-tuned via
        # ctx_headroom.py anyway, so leave sizing to the profile.
        -fit "${LLAMA_FIT:-off}"
        -c "$LLAMA_CONTEXT"
        -t "$LLAMA_THREADS"
        -tb "$LLAMA_BATCH_THREADS"
        --parallel "$LLAMA_PARALLEL"
        --cache-type-k "$LLAMA_CACHE_TYPE_K"
        --cache-type-v "$LLAMA_CACHE_TYPE_V"
        --flash-attn "$LLAMA_FLASH_ATTN"
        --ubatch-size "$LLAMA_UBATCH"
    )

    if [[ -n "${LLAMA_CACHE_RAM:-}" ]]; then
        LLAMA_CMD+=(--cache-ram "$LLAMA_CACHE_RAM")
    fi

    if [[ "$LLAMA_CONTEXT_SHIFT" == "1" ]]; then
        LLAMA_CMD+=(--context-shift)
    else
        LLAMA_CMD+=(--no-context-shift)
    fi

    case "$LLAMA_SPEC_MODE" in
        mtp)
            LLAMA_CMD+=(
                --spec-type draft-mtp
                --spec-draft-n-max "${LLAMA_SPEC_DRAFT_N_MAX:-2}"
                --spec-draft-p-min "${LLAMA_SPEC_DRAFT_P_MIN:-0.0}"
            )
            if [[ -n "${LLAMA_SPEC_DRAFT_TYPE_K:-}" ]]; then
                LLAMA_CMD+=(--spec-draft-type-k "$LLAMA_SPEC_DRAFT_TYPE_K")
            fi
            if [[ -n "${LLAMA_SPEC_DRAFT_TYPE_V:-}" ]]; then
                LLAMA_CMD+=(--spec-draft-type-v "$LLAMA_SPEC_DRAFT_TYPE_V")
            fi
            ;;
        dflash)
            LLAMA_CMD+=(
                --spec-type "${LLAMA_DFLASH_SPEC_TYPE:-dflash}"
                --spec-draft-model "$LLAMA_DRAFTER"
                --spec-draft-ngl "${LLAMA_SPEC_DRAFT_NGL:-99}"
            )
            if [[ -n "${LLAMA_SPEC_DRAFT_N_MAX:-}" ]]; then
                LLAMA_CMD+=(--spec-draft-n-max "$LLAMA_SPEC_DRAFT_N_MAX")
            fi
            if [[ -n "${LLAMA_SPEC_DRAFT_N_MIN:-}" ]]; then
                LLAMA_CMD+=(--spec-draft-n-min "$LLAMA_SPEC_DRAFT_N_MIN")
            fi
            if [[ -n "${LLAMA_SPEC_DFLASH_CROSS_CTX:-}" ]]; then
                LLAMA_CMD+=(--spec-dflash-cross-ctx "$LLAMA_SPEC_DFLASH_CROSS_CTX")
            fi
            # The drafter keeps its own KV cache, and its precision affects how
            # often draft and target agree -- i.e. acceptance, which is what
            # actually sets speculative decode speed. Measured on Muse Glimmer:
            # moving the TARGET KV q8_0 -> f16 raised DFlash acceptance 43.6% ->
            # 58.5% and decode 28.0 -> 33.2 t/s. These flags were previously only
            # emitted on the mtp path, so the dflash drafter's KV type could not
            # be set from a profile at all.
            if [[ -n "${LLAMA_SPEC_DRAFT_TYPE_K:-}" ]]; then
                LLAMA_CMD+=(--spec-draft-type-k "$LLAMA_SPEC_DRAFT_TYPE_K")
            fi
            if [[ -n "${LLAMA_SPEC_DRAFT_TYPE_V:-}" ]]; then
                LLAMA_CMD+=(--spec-draft-type-v "$LLAMA_SPEC_DRAFT_TYPE_V")
            fi
            ;;
    esac

    LLAMA_CMD+=(
        --temp "$LLAMA_TEMPERATURE"
        --top-p "$LLAMA_TOP_P"
        --top-k "$LLAMA_TOP_K"
        --min-p "$LLAMA_MIN_P"
        --presence-penalty "$LLAMA_PRESENCE_PENALTY"
        --repeat-penalty "$LLAMA_REPEAT_PENALTY"
        --reasoning "$LLAMA_REASONING"
    )

    if [[ -n "$LLAMA_CHAT_TEMPLATE_KWARGS" ]]; then
        LLAMA_CMD+=(--chat-template-kwargs "$LLAMA_CHAT_TEMPLATE_KWARGS")
    fi
    if [[ -n "$LLAMA_API_KEY_FILE" ]]; then
        LLAMA_CMD+=(--api-key-file "$LLAMA_API_KEY_FILE")
    fi
    if [[ "$LLAMA_METRICS" == "1" ]]; then
        LLAMA_CMD+=(--metrics)
    fi

    LLAMA_CMD+=(
        --port "$LLAMA_PORT"
        --host "$LLAMA_HOST"
        --timeout "$LLAMA_TIMEOUT"
        --log-colors off
    )
    if [[ "$LLAMA_JINJA" == "1" ]]; then
        LLAMA_CMD+=(--jinja)
    else
        LLAMA_CMD+=(--no-jinja)
    fi

    LLAMA_CMD+=("${LLAMA_EXTRA_ARGS[@]}")
}

llama_print_command() {
    printf '%q ' "${LLAMA_CMD[@]}"
    printf '\n'
}
