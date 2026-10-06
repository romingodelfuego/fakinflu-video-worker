# fakinflu — worker RunPod : video (Wan 2.2 Fun Control 14B via ComfyUI, modeles BAKES)
# Build par RunPod (integration GitHub) a chaque GitHub Release : make worker-release W=video
#
# Base : runpod/worker-comfyui, meme strategie de tag epingle que comfyui/serverless.
# 5.10.0-base embarque ComfyUI 0.34.0 (torch 2.11 cu128, ffmpeg, PyAV) : tous les nodes du
# graphe y sont NATIFS (Wan22FunControlToVideo, LoadVideo, GetVideoComponents, Canny,
# CreateVideo, SaveVideo, KSamplerAdvanced, ModelSamplingSD3, LoraLoaderModelOnly...).
# Aucun custom node -> pas de reinstall de ComfyUI ici (contrairement au worker Qwen 2.1).
ARG WORKER_TAG=5.10.0-base
FROM runpod/worker-comfyui:${WORKER_TAG}

# Garde-fou au build : echoue ICI si la version de ComfyUI n'a pas les nodes Wan 2.2 Fun.
RUN grep -q '"Wan22FunControlToVideo"' /comfyui/comfy_extras/nodes_wan.py \
 && grep -q '"LoadVideo"' /comfyui/comfy_extras/nodes_video.py \
 && grep -q '"Canny"' /comfyui/comfy_extras/nodes_canny.py

# --- Poids BAKES (aucun volume, aucun telechargement au cold start) ----------
# 14B fp8 (2 experts MoE) + umt5 fp8 + VAE 2.1 + LoRA 4 pas : ~38 Go.
# WITH_5B=1 ajoute Fun Control 5B + VAE 2.2 (~11,4 Go) pour des previews moins cheres.
ARG WITH_5B=0
# aria2 : telechargements multi-connexions (cf. download_models.sh, limite de build de 30 min)
RUN apt-get update && apt-get install -y --no-install-recommends aria2 \
 && rm -rf /var/lib/apt/lists/*
COPY download_models.sh /tmp/download_models.sh
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh high
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh low
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh common
RUN if [ "${WITH_5B}" = "1" ]; then MODELS_DIR=/comfyui/models bash /tmp/download_models.sh 5b; fi \
 && rm -f /tmp/download_models.sh

# --- Handler fakinflu (remplace /handler.py de worker-comfyui) ---------------
# /start.sh (herite) lance ComfyUI en arriere-plan puis `python -u /handler.py`.
# Notre handler accepte le contrat fakinflu ⑥ (prompt, control video, refs) au lieu de
# {"workflow", "images"} : il construit le graphe depuis workflow_template*.json.
COPY workflow_template.json workflow_template_5b.json /fakinflu/
COPY handler.py /handler.py
COPY test_input.json /test_input.json
ENV FAKINFLU_WORKFLOW_DIR=/fakinflu \
    COMFY_LOG_LEVEL=INFO

CMD ["/start.sh"]
