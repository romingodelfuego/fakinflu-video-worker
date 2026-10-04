# Worker RunPod : video (⑥ vidéo finale, Wan 2.2 Fun Control)

Dépôt publié : https://github.com/romingodelfuego/fakinflu-video-worker. Ce dépôt est un
**miroir** : la source fait foi dans `video/serverless/` de fakinflu.

Le worker transforme le **blocking 3D** du staging (`pose.mp4` / `depth.mp4` / `clay.mp4`)
plus les **images de référence** des personnages et un prompt texte en une **vidéo réaliste**.
La 3D décide qui, où, quand et la caméra ; le modèle vidéo apporte le réalisme.

## Approche retenue : Wan 2.2 Fun Control 14B (fp8), nodes ComfyUI natifs

| | |
|---|---|
| Modèle | `alibaba-pai/Wan2.2-Fun-A14B-Control` (MoE 2 experts high/low noise), poids fp8 repackagés `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (épinglés sur un commit, sha256 vérifiés) |
| Graphe | `workflow_template.json`, converti du template officiel ComfyUI `video_wan2_2_14B_fun_control` |
| Nodes | **100 % natifs** dans ComfyUI 0.34.0 : `Wan22FunControlToVideo`, `LoadVideo`, `GetVideoComponents`, `Canny`, `KSamplerAdvanced` ×2, `ModelSamplingSD3`, `LoraLoaderModelOnly`, `CreateVideo`, `SaveVideo`… **aucun custom node** |
| Base | `runpod/worker-comfyui:5.10.0-base` (ComfyUI 0.34.0, torch 2.11 cu128, ffmpeg) : pas de réinstallation de ComfyUI |
| Rapide | LoRA lightx2v 4 pas (`wan2.2_i2v_lightx2v_4steps_lora_v1_*`, Apache-2.0), utilisée par le template officiel |

**Pourquoi ce choix**
- Fun Control est **entraîné** sur des vidéos de contrôle pose (squelette façon OpenPose), depth et
  canny : c'est exactement ce que sort le staging. Le rendu `clay` (gris uni) n'est pas une modalité
  d'entraînement : le graphe le passe dans le node `Canny` natif (arêtes), ce qui garde la silhouette
  et la caméra.
- `Wan22FunControlToVideo` accepte une **image de référence** (`ref_image`) en plus de la vidéo de
  contrôle. ComfyUI l'injecte en `reference_latents` (référence d'apparence, **pas** la première
  image) : la mise en page de la référence n'impose donc pas la composition du plan.
- Il existe un template officiel Comfy-Org, donc un graphe déjà validé, avec ses réglages (shift 8,
  bascule high→low à mi-parcours, 20 pas cfg 3,5 ou 4 pas cfg 1 avec LoRA).
- Aucune dépendance fragile : pas de pack de custom nodes à épingler. Le pré-traitement
  (ré-échantillonnage, recadrage) est fait par ffmpeg dans le handler.

**Plusieurs personnages.** Le node n'accepte qu'**une** image de référence. Le handler compose donc
une **planche** (`ref_mode: "sheet"`, par défaut) : les personnages côte à côte sur fond blanc, à la
taille de la vidéo. `first` n'envoie que le premier personnage, `none` aucune ref (l'apparence vient
alors du prompt seul).

### Alternatives écartées (pour l'instant)

| Option | Pour | Contre |
|---|---|---|
| **Wan 2.2 VACE-Fun 14B** (`wan2.2_fun_vace_*`, node natif `WanVaceToVideo`) | référence d'identité + contrôle, masques | pas de template officiel 2.2 dans ComfyUI, poids plus lourds (17,4 Go/expert fp8), une seule `reference_image` également. **Plan B** si l'identité tient mal : même structure de graphe, à essayer. |
| **Wan 2.2 Animate 14B** (`WanAnimateToVideo`) | très bonne identité, transfert de mouvement | pensé pour **un** personnage, demande une vraie vidéo source (crops visage, masques), pré-traitement par custom nodes |
| **WanAnimate2ToVideo** (ComfyUI 0.34) | pose + référence | node **expérimental**, un seul personnage |
| **Fun Control 5B** | ~3× moins cher | qualité nettement inférieure (le template officiel le dit). Gardé en **option preview** : build `WITH_5B=1`, puis `model: "5b"` |

## Contrat d'API (scene/CONTRACT.md §⑥)

```json
{"input": {
  "prompt": "…", "negative": "…",
  "control": {"type": "pose|depth|clay", "video": "<b64 mp4>"},
  "refs": [{"actor": "A", "image": "<b64>"}],
  "width": 480, "height": 832, "fps": 16, "seed": 0,
  "model": "14b", "preset": "fast", "max_frames": 81, "ref_mode": "sheet",
  "overrides": {"sampler_high.steps": 6}
}}
→ {"video": "<b64 mp4>" | "<url>", "stats": {"frames": 81, "duration_s": 5.06, "comfy_s": …, "gpu": …, "vram_used_mb_after": …}}
```

- Les champs après `seed` sont des **options fakinflu** (facultatives).
- `overrides` modifie un champ scalaire du graphe (`<id du node>.<clé>`), pour régler sans rebuild.
- Lot possible : `{"input": {"items": [{…, "id": "a"}]}}` → `{"results": [{"id": "a", "video": …}]}`.
- Une entrée invalide renvoie `{"error": "entree invalide : …"}` sans faire échouer le job RunPod.

**Pourquoi un handler maison** plutôt que le format natif de worker-comfyui
(`{"workflow", "images"}`) :
- le contrat ⑥ est respecté tel quel, et le client reste trivial ;
- le graphe est **versionné avec l'image**, donc toujours cohérent avec les poids bakés ;
- le handler natif ne sait pas déposer une **vidéo** dans `input/` ni renvoyer un mp4 proprement.

`/start.sh` (hérité) lance ComfyUI puis `python -u /handler.py`, qui est le nôtre.

**Traitement d'un job**
1. Ré-échantillonnage de la vidéo de contrôle au `fps` cible (ffmpeg), recadrage « cover » en
   `width`×`height`, coupe à `max_frames`. La longueur est ramenée à **4k+1** images.
2. Composition de la planche de référence (PIL).
3. Graphe rempli depuis le bloc `_fakinflu` du template, envoyé à ComfyUI, puis polling `/history`.
4. Le mp4 revient en base64. Au-delà de ~7 Mo (`VIDEO_MAX_INLINE_MB`), il est ré-encodé (crf 28).
   Si `BUCKET_ENDPOINT_URL` (et ses identifiants) est défini sur l'endpoint, il est uploadé et on
   renvoie une URL.

**Conventions d'entrée attendues du staging (④)**
- `pose` : squelette coloré façon OpenPose/DWPose sur **fond noir** (format d'entraînement de Fun Control).
- `depth` : **proche = clair**, loin = sombre (convention Depth-Anything / MiDaS).
- 16 fps et ≤ 81 images (~5 s) par appel. Une scène plus longue se génère en plusieurs plans
  (`--start` côté client). La continuité entre plans n'est pas garantie : il faut couper sur un
  changement de caméra.

## VRAM, GPU et coûts

- **Poids bakés : ~38 Go** (2 × 14,3 Go experts fp8, umt5 fp8 6,7 Go, LoRA 2 × 1,2 Go, VAE 0,25 Go).
  Image ≈ 53 Go avec la base (≈ 64 Go avec `WITH_5B=1`), sous la limite de 80 Go.
- ComfyUI ne garde **qu'un expert** en VRAM à la fois (~14 Go), plus umt5 (~7 Go, déchargé après
  l'encodage) et les activations. En 480p (480×832×81), on estime ~20-28 Go au pic, donc **48 Go**
  donnent une marge confortable, 720p compris.
- Une carte 24 Go passerait en 480p (cas courant sur RTX 4090), mais à la limite, avec du
  déchargement. Non retenu : le risque d'OOM coûte plus cher que l'écart de prix.
- Pools : `AMPERE_48` (A6000 / A40, **1,22 $/h** serverless), puis `ADA_48_PRO` (L40S / 6000 Ada,
  **1,75 $/h**, calcul fp8 natif donc plus rapide). Ce sont les prix du compte, relevés le 2026-10-04.
- Pour du 720p en preset `quality`, on peut passer sur `AMPERE_80` (A100, 2,72 $/h).

**Estimation par clip de 5 s (81 images, 480×832), NON MESURÉE**

Ordres de grandeur extrapolés des temps publics de Wan 14B ; à remplacer par les `stats` réelles
après le premier run.

| Preset | Pas | A6000 (1,22 $/h) | L40S (1,75 $/h) |
|---|---|---|---|
| `fast` (LoRA 4 pas, cfg 1) | 4 | ~2-4 min → **~0,04-0,08 $** | ~1,5-2,5 min → ~0,05-0,07 $ |
| `quality` (20 pas, cfg 3,5) | 20 | ~12-18 min → **~0,25-0,37 $** | ~7-10 min → ~0,20-0,30 $ |
| cold start (chargement ~38 Go) | – | +1-3 min → +0,02-0,06 $ | idem |

Le 720p coûte environ 2,5× le 480p. Pour limiter les cold starts, enchaîne les plans en rafale,
worker chaud.

## Réglages de l'endpoint (`worker.json`)

- `min 0`, `max 1`, `idleTimeout 5 s`, FlashBoot activé.
- Timeout de **30 min** : le preset `quality` sur A6000 dépasse les 20 min.
- Container disk de 20 Go (les poids sont dans l'image).
- Aucun secret dans l'image ; un éventuel bucket passe par les variables d'environnement de l'endpoint.

## Limites connues

- **Identité multi-personnages.** La planche donne au modèle l'apparence (vêtements, couleurs,
  coiffure), mais rien ne lie explicitement « personnage A de la planche » au squelette de gauche.
  Avec deux personnes, il peut **échanger ou mélanger** les identités. Pour limiter le risque :
  décrire chaque personne dans `video.prompt` (position, vêtements) ; garder des tenues très
  contrastées ; au besoin, utiliser `ref_mode: "first"` pour le personnage principal. Les visages
  restent approximatifs (ce n'est pas un modèle d'ID facial).
- **Durée** : ~5 s par appel (81 images à 16 fps). Au-delà, l'enchaînement de plans n'est pas
  cohérent d'un appel à l'autre (seed et référence identiques, mais pas de continuité de frame).
- **Contacts** (gifle) : le contrôle pose donne le geste, pas l'impact. Le prompt doit le décrire.
- Le rendu `clay`, passé en Canny, contraint fortement les contours. Les vêtements amples peuvent
  ne pas suivre la silhouette du mannequin.
- La LoRA 4 pas réduit la dynamique du mouvement (note du template officiel) : `quality` pour le rendu final.

## Licences

- Wan 2.2 et Wan2.2-Fun-A14B/5B-Control : **Apache-2.0** (alibaba-pai, Wan-AI).
- Repackaging Comfy-Org (`Comfy-Org/Wan_2.2_ComfyUI_Repackaged`) : Apache-2.0.
- LoRA lightx2v (`lightx2v/Wan2.2-Lightning`) : Apache-2.0.
- umt5-xxl (texte) : Apache-2.0. ComfyUI : GPL-3.0 (exécuté tel quel dans l'image, non redistribué
  modifié). La vidéo générée n'est soumise à aucune restriction de licence du modèle. La
  responsabilité du contenu (personnes réelles en référence) reste à l'utilisateur.

## Cycle de vie

```bash
make worker-status W=video                 # local vs dernière release (0 crédit)
make worker-release W=video DRY=1          # ce qui partirait, sans rien publier
make worker-release W=video BUMP=minor NOTES="..."   # tag vX.Y.Z + GitHub Release → rebuild RunPod
```

**Première release seulement** : lier le dépôt dans la console RunPod (**Serverless → New
Endpoint → Import Git Repository**), appliquer `worker.json`, noter l'ID dans `.env` →
`RUNPOD_VIDEO_ENDPOINT_ID`. Build args : `WORKER_TAG=5.10.0-base`, `WITH_5B=0|1`. Le build
télécharge ~38 Go (~50 Go avec le 5B) côté RunPod.

Test local du handler, sur une machine GPU avec l'image : `python /handler.py --test_input "$(cat test_input.json)"`
(ComfyUI doit tourner : `/start.sh` le lance).

Côté client : `make -C video dry SCENE=data/out/scenes/<id>` (0 crédit), puis `make -C video video SCENE=…`.
