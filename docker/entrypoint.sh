#!/usr/bin/env bash
# T2Music entrypoint — routes to sub-commands via first arg
set -euo pipefail

CMD="${1:-help}"
shift || true

case "$CMD" in
  help)
    echo "T2Music Phase A CLI"
    echo ""
    echo "Commands:"
    echo "  train-yue2       Launch 8-card YuE2-3B training (FSDP)"
    echo "  train-acestep    Launch 8-card ACE-Step v1 training"
    echo "  infer-yue2       Run YuE2-3B inference on prompts file"
    echo "  infer-acestep    Run ACE-Step v1 inference on prompts file"
    echo "  eval-fad         Compute FAD on a dir of audio"
    echo "  eval-clap        Compute CLAP score"
    echo "  eval-all         Run all evaluations"
    echo "  data-fingerprint Compute dataset fingerprint"
    echo "  download-musiccaps  Download MusicCaps subset"
    echo "  bash             Drop into shell"
    ;;

  train-yue2)
    torchrun --nproc_per_node=8 \
        /workspace/src/train/main.py \
        --config /workspace/configs/yue2_3b.yaml \
        "$@"
    ;;

  train-acestep)
    torchrun --nproc_per_node=8 \
        /workspace/src/train/main.py \
        --config /workspace/configs/acestep_v1.yaml \
        "$@"
    ;;

  infer-yue2)
    python /workspace/src/train/infer_yue2.py "$@"
    ;;

  infer-acestep)
    python /workspace/src/train/infer_acestep.py "$@"
    ;;

  eval-fad)
    python /workspace/src/eval/compute_fad.py "$@"
    ;;

  eval-clap)
    python /workspace/src/eval/compute_clap.py "$@"
    ;;

  eval-all)
    python /workspace/src/eval/compute_fad.py "$@"
    python /workspace/src/eval/compute_clap.py "$@"
    ;;

  data-fingerprint)
    python /workspace/scripts/compute_data_fingerprint.py "$@"
    ;;

  download-musiccaps)
    bash /workspace/scripts/download_datasets.sh musiccaps "$@"
    ;;

  bash)
    exec /bin/bash
    ;;

  *)
    echo "Unknown command: $CMD"
    echo "Run 'help' for usage."
    exit 1
    ;;
esac
