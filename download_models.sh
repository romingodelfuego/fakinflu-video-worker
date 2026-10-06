#!/usr/bin/env bash
# fakinflu — telecharge les poids Wan 2.2 Fun Control dans un arbre ComfyUI/models.
# Depot : Comfy-Org/Wan_2.2_ComfyUI_Repackaged (Apache-2.0), EPINGLE sur un commit,
# chaque fichier verifie par son sha256 (= oid LFS lu via l'API HF).
# aria2c (installe par le Dockerfile) : 16 connexions paralleles + reprise. Le build RunPod est
# limite a 30 min ; en v1.0.0 un expert de 14 Go a mis plus de 25 min en wget mono-connexion.
# Repli sur wget (present dans l'image worker-comfyui ; curl ne l'est PAS).
#
# Usage (un groupe par couche Docker -> couches ~14 Go, pull en parallele) :
#   MODELS_DIR=/comfyui/models ./download_models.sh high   # expert « high noise » 14B fp8 (14,3 Go)
#   MODELS_DIR=/comfyui/models ./download_models.sh low    # expert « low noise » 14B fp8  (14,3 Go)
#   MODELS_DIR=/comfyui/models ./download_models.sh common # umt5 fp8 + VAE 2.1 + LoRA lightx2v 4 pas (9,4 Go)
#   MODELS_DIR=/comfyui/models ./download_models.sh 5b     # option : Fun Control 5B + VAE 2.2 (11,4 Go)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/comfyui/models}"
REV="ee6f4a40737a995bf5818954cfce6d59443b0f04"   # Comfy-Org/Wan_2.2_ComfyUI_Repackaged, 2026-09-23
BASE="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/${REV}/split_files"

# DL <sous-dossier> <fichier> <sha256>
DL() {
  local sub="$1" name="$2" sha="$3" dest="$MODELS_DIR/$1/$2"
  mkdir -p "$MODELS_DIR/$sub"
  echo ">> $sub/$name"
  local t0=$SECONDS
  if command -v aria2c >/dev/null; then
    aria2c -q -c -x 16 -s 16 -k 64M --max-tries=10 --retry-wait=5 --timeout=30 \
           --lowest-speed-limit=1M --file-allocation=none --console-log-level=warn \
           -d "$MODELS_DIR/$sub" -o "$name" "$BASE/$sub/$name"
  else
    wget -q -c --tries=10 --timeout=30 --waitretry=5 -O "$dest" "$BASE/$sub/$name"
  fi
  echo "   telecharge en $((SECONDS - t0)) s"
  echo "$sha  $dest" | sha256sum -c --quiet - || { echo "!! sha256 invalide : $dest"; rm -f "$dest"; exit 1; }
}

case "${1:-}" in
  high)
    DL diffusion_models wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors \
       aa2f6b6f4cfc8a75a273a075db31730530f93320a996b153b8825205c3a3c498 ;;
  low)
    DL diffusion_models wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors \
       0f83c7b1cd6d509ff3504a8f604ca9d9876a8b1ed04f6e05095c4a7ccb9009cd ;;
  common)
    DL text_encoders umt5_xxl_fp8_e4m3fn_scaled.safetensors \
       c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68
    DL vae wan_2.1_vae.safetensors \
       2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b
    # LoRA de distillation 4 pas (lightx2v / Wan2.2-Lightning, Apache-2.0) : mode « fast »
    DL loras wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
       d176c808d6fc461999b68e321efcb7501b20b8c3797523ed0df14f7d1deff11e
    DL loras wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
       024f21de095bc8fad9809ded3e9e49a2e170dcf27075da8145ba7d60d8aab7f9 ;;
  5b)
    DL diffusion_models wan2.2_fun_control_5B_bf16.safetensors \
       ace4718a7c87ee3e5606a68ab79142c4395e81aece76b8120bc886f0fbbe1d16
    DL vae wan2.2_vae.safetensors \
       e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156 ;;
  *)
    echo "usage : $0 high|low|common|5b"; exit 1 ;;
esac

echo ">> OK :"
find "$MODELS_DIR" -maxdepth 2 -type f -name '*.safetensors' -printf '   %p  (%s octets)\n'
