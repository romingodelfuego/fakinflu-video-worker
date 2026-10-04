#!/usr/bin/env python3
"""fakinflu — worker RunPod : video (Wan 2.2 Fun Control via ComfyUI).

Remplace le /handler.py de runpod/worker-comfyui. /start.sh lance ComfyUI (127.0.0.1:8188)
en arriere-plan, puis ce script. On expose le CONTRAT fakinflu ⑥ (scene/CONTRACT.md) au lieu
du format natif {"workflow", "images"} : le graphe est construit ICI depuis
workflow_template*.json, ce qui garde le client simple et le graphe versionne avec l'image.

Entree (a plat dans "input", ou un lot dans "input.items") :
  {"prompt": "...", "negative": "...",
   "control": {"type": "pose|depth|clay", "video": "<b64 mp4>"},
   "refs": [{"actor": "A", "image": "<b64>"}],
   "width": 480, "height": 832, "fps": 16, "seed": 0,
   # optionnels fakinflu :
   "model": "14b|5b", "preset": "fast|quality", "max_frames": 81,
   "ref_mode": "sheet|first|none", "overrides": {"<node>.<cle>": valeur}}
Sortie : {"video": "<b64 mp4>" | "<url>", "stats": {...}}  (lot : {"results": [...]})

Pre-traitement (ffmpeg, present dans l'image) : la video de controle est re-echantillonnee
au fps cible, recadree « cover » en width x height et coupee a max_frames ; la longueur est
ramenee a 4k+1 images (contrainte du VAE Wan). Plusieurs refs -> une PLANCHE unique
(personnages cote a cote sur fond blanc), car Wan22FunControlToVideo n'accepte qu'une image.
"""
import base64, io, json, os, subprocess, time, traceback, urllib.error, urllib.request, uuid

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")
WF_DIR = os.environ.get("FAKINFLU_WORKFLOW_DIR", os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = {"14b": "workflow_template.json", "5b": "workflow_template_5b.json"}
CONTROL_TYPES = ("pose", "depth", "clay")
MAX_INLINE = int(float(os.environ.get("VIDEO_MAX_INLINE_MB", "7")) * 1024 * 1024)
JOB_TIMEOUT = int(os.environ.get("VIDEO_JOB_TIMEOUT_S", "1700"))


class InputError(ValueError):
    """Entree invalide (renvoyee telle quelle au client)."""


# ---------------------------------------------------------------- utilitaires

def _b64decode(s):
    if not isinstance(s, str) or not s:
        raise InputError("champ base64 vide")
    if s.strip().startswith("data:") and "," in s[:100]:
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def _run(cmd, timeout=300):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} : {(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout


def _mult16(v, name):
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise InputError(f"{name} doit etre un entier")
    if not 256 <= v <= 1280:
        raise InputError(f"{name}={v} hors bornes [256, 1280]")
    return v // 16 * 16


def load_template(model):
    if model not in TEMPLATES:
        raise InputError(f"model inconnu : {model} (attendu : {', '.join(TEMPLATES)})")
    with open(os.path.join(WF_DIR, TEMPLATES[model])) as f:
        wf = json.load(f)
    cfg = wf.get("_fakinflu") or {}
    for k in [k for k in wf if k.startswith("_")]:
        wf.pop(k)
    return wf, cfg


def model_available(wf):
    """Vrai si tous les poids references par le graphe sont presents (ex. 5B optionnel)."""
    where = {"UNETLoader": ("diffusion_models", "unet_name"), "VAELoader": ("vae", "vae_name"),
             "CLIPLoader": ("text_encoders", "clip_name"), "LoraLoaderModelOnly": ("loras", "lora_name")}
    for n in wf.values():
        if n["class_type"] in where:
            sub, key = where[n["class_type"]]
            if not os.path.isfile(os.path.join(COMFY_DIR, "models", sub, n["inputs"][key])):
                return False
    return True


def set_path(wf, path, value):
    """'node.cle' -> wf[node]['inputs'][cle] = value (la cle peut contenir des points)."""
    node, key = path.split(".", 1)
    if node not in wf or key not in wf[node]["inputs"]:
        raise InputError(f"override inconnu : {path}")
    old = wf[node]["inputs"][key]
    if isinstance(old, list) or isinstance(value, (list, dict)):
        raise InputError(f"override {path} : seules les valeurs scalaires sont modifiables")
    wf[node]["inputs"][key] = value


# ---------------------------------------------------------------- pre-traitement

def normalize_control(src, dst, width, height, fps, max_frames):
    """Re-echantillonne / recadre / coupe la video de controle. Renvoie le nb d'images."""
    vf = (f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase:flags=bilinear,"
          f"crop={width}:{height},setsar=1,format=yuv420p")
    _run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", vf, "-frames:v", str(max_frames),
          "-an", "-c:v", "libx264", "-crf", "14", "-preset", "veryfast", dst])
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", dst])
    return int(out.strip().split(",")[0])


def build_ref_sheet(images, width, height, mode="sheet"):
    """Planche de reference width x height : personnages cote a cote sur fond blanc."""
    from PIL import Image
    imgs = []
    for raw in images:
        im = Image.open(io.BytesIO(raw))
        im.load()
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        imgs.append(im.convert("RGB"))
    if mode == "first":
        imgs = imgs[:1]
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    n = len(imgs)
    slot_w = width / n
    for i, im in enumerate(imgs):
        scale = min(slot_w * 0.96 / im.width, height * 0.96 / im.height)
        w, h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        im = im.resize((w, h), Image.LANCZOS)
        x = int(i * slot_w + (slot_w - w) / 2)
        y = (height - h) // 2
        sheet.paste(im, (x, y))
    return sheet


# ---------------------------------------------------------------- graphe

def build_workflow(item, control_name, ref_name, n_frames):
    """Construit le graphe API a partir du template + de l'item (deja valide)."""
    wf, cfg = load_template(item["model"])
    d = cfg.get("defaults", {})

    def put(spec, value):
        wf[spec["node"]]["inputs"][spec["key"]] = value

    put(cfg["prompt"], item["prompt"])
    if item.get("negative"):
        put(cfg["negative"], item["negative"])
    put(cfg["control"], control_name)
    put(cfg["fps"], float(item["fps"]))
    put(cfg["save"], f"fakinflu/{item['_job']}")
    for s in cfg.get("seed", []):
        put(s, int(item["seed"]))

    length = min(n_frames, int(item.get("max_frames") or d.get("max_frames", 81)))
    length = (length - 1) // 4 * 4 + 1
    if length < 5:
        raise InputError(f"video de controle trop courte ({n_frames} images apres normalisation, min 5)")
    cond = wf[cfg["cond"]]["inputs"]
    cond.update(width=item["width"], height=item["height"], length=length)

    # clay (rendu gris) n'est pas une modalite d'entrainement de Fun Control -> aretes Canny
    if item["control"]["type"] == "clay":
        cond["control_video"] = [cfg["canny"], 0]
    else:
        wf.pop(cfg["canny"], None)

    if ref_name:
        put(cfg["ref"], ref_name)
    else:
        wf.pop(cfg["ref"]["node"], None)
        cond.pop("ref_image", None)

    preset = item.get("preset") or d.get("preset", "fast")
    if preset not in cfg.get("presets", {}):
        raise InputError(f"preset inconnu : {preset} (attendu : {', '.join(cfg.get('presets', {}))})")
    for path, v in cfg["presets"][preset].items():
        set_path(wf, path, v)
    for path, v in (item.get("overrides") or {}).items():
        set_path(wf, path, v)
    return wf, {"length": length, "preset": preset}


def validate(item):
    """Normalise l'item du contrat ⑥ ; leve InputError si invalide."""
    if not isinstance(item, dict):
        raise InputError("item : objet JSON attendu")
    it = dict(item)
    if not isinstance(it.get("prompt"), str) or not it["prompt"].strip():
        raise InputError("prompt requis")
    ctrl = it.get("control") or {}
    if ctrl.get("type", "pose") not in CONTROL_TYPES:
        raise InputError(f"control.type : {ctrl.get('type')} (attendu : {', '.join(CONTROL_TYPES)})")
    if not ctrl.get("video"):
        raise InputError("control.video (mp4 en base64) requis")
    it["control"] = {"type": ctrl.get("type", "pose"), "video": ctrl["video"]}
    it["model"] = str(it.get("model") or "14b").lower()
    _, cfg = load_template(it["model"])
    d = cfg.get("defaults", {})
    it["width"] = _mult16(it.get("width", 480), "width")
    it["height"] = _mult16(it.get("height", 832), "height")
    it["fps"] = int(it.get("fps") or d.get("fps", 16))
    if not 4 <= it["fps"] <= 30:
        raise InputError("fps hors bornes [4, 30]")
    seed = it.get("seed")
    it["seed"] = int(seed) if seed is not None else int.from_bytes(os.urandom(6), "big")
    refs = it.get("refs") or []
    if not isinstance(refs, list) or any(not isinstance(r, dict) or not r.get("image") for r in refs):
        raise InputError("refs : liste de {actor, image(b64)} attendue")
    if len(refs) > 4:
        raise InputError("refs : 4 personnages maximum")
    it["refs"] = refs
    it["ref_mode"] = it.get("ref_mode") or "sheet"
    if it["ref_mode"] not in ("sheet", "first", "none"):
        raise InputError("ref_mode : sheet|first|none")
    if it.get("max_frames") is not None:
        mf = int(it["max_frames"])
        if not 5 <= mf <= 161:
            raise InputError("max_frames hors bornes [5, 161]")
        it["max_frames"] = mf
    return it


# ---------------------------------------------------------------- ComfyUI

def _http(method, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def wait_comfy(max_s=600):
    t = time.time()
    while time.time() - t < max_s:
        try:
            _http("GET", "/system_stats", timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(1)
    raise RuntimeError(f"ComfyUI injoignable sur {COMFY} apres {max_s}s")


def queue_and_wait(wf, client_id):
    try:
        r = _http("POST", "/prompt", {"prompt": wf, "client_id": client_id})
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:1500]
        raise RuntimeError(f"ComfyUI a refuse le graphe (HTTP {e.code}) : {body}")
    pid = r.get("prompt_id")
    if not pid:
        raise RuntimeError(f"pas de prompt_id : {json.dumps(r)[:500]}")
    t0 = time.time()
    while True:
        if time.time() - t0 > JOB_TIMEOUT:
            try:
                _http("POST", "/interrupt", {})
            except Exception:
                pass
            raise RuntimeError(f"timeout ComfyUI ({JOB_TIMEOUT}s)")
        time.sleep(2)
        h = _http("GET", f"/history/{pid}")
        if pid not in h:
            continue
        entry = h[pid]
        st = entry.get("status") or {}
        if st.get("status_str") == "error":
            msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
            detail = msgs[-1][1] if msgs else st
            if isinstance(detail, dict):
                detail = {k: detail.get(k) for k in ("node_id", "node_type", "exception_message")}
            raise RuntimeError(f"erreur d'execution ComfyUI : {json.dumps(detail, ensure_ascii=False)[:800]}")
        if st.get("completed") or entry.get("outputs"):
            return entry.get("outputs") or {}, time.time() - t0


def find_video(outputs):
    for node_out in outputs.values():
        for key in ("images", "videos", "gifs", "animated"):
            for f in node_out.get(key) or []:
                if isinstance(f, dict) and str(f.get("filename", "")).endswith((".mp4", ".webm", ".mkv")):
                    return os.path.join(COMFY_DIR, f.get("type", "output"), f.get("subfolder", ""), f["filename"])
    raise RuntimeError(f"aucune video dans les sorties ComfyUI : {json.dumps(outputs)[:500]}")


def gpu_info():
    try:
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                    "--format=csv,noheader,nounits"], timeout=10).strip().splitlines()[0]
        name, total, used = [s.strip() for s in out.split(",")]
        return {"gpu": name, "vram_total_mb": int(total), "vram_used_mb_after": int(used)}
    except Exception:
        return {}


def deliver(path, job_id):
    """b64 inline (re-encode si trop gros), ou URL si un bucket S3 est configure."""
    if os.environ.get("BUCKET_ENDPOINT_URL"):
        from runpod.serverless.utils import rp_upload
        return rp_upload.upload_file_to_bucket(f"{job_id}.mp4", path), os.path.getsize(path)
    size = os.path.getsize(path)
    if size * 4 / 3 > MAX_INLINE:
        smaller = path[:-4] + "_small.mp4"
        _run(["ffmpeg", "-y", "-v", "error", "-i", path, "-c:v", "libx264", "-crf", "28",
              "-preset", "medium", "-pix_fmt", "yuv420p", "-an", smaller])
        path, size = smaller, os.path.getsize(smaller)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), size


# ---------------------------------------------------------------- job

def process(item, job_id):
    t_start = time.time()
    it = validate(item)
    tag = f"fk_{job_id}_{uuid.uuid4().hex[:6]}"
    it["_job"] = tag
    in_dir = os.path.join(COMFY_DIR, "input")
    os.makedirs(in_dir, exist_ok=True)
    raw_ctrl = os.path.join(in_dir, tag + "_raw.mp4")
    ctrl_name = tag + "_control.mp4"
    ref_name = None
    tmp = [raw_ctrl, os.path.join(in_dir, ctrl_name)]
    try:
        wf_check, cfg = load_template(it["model"])
        if not model_available(wf_check):
            raise InputError(f"modele {it['model']} absent de cette image (build WITH_5B=1 pour le 5B)")
        max_frames = int(it.get("max_frames") or cfg.get("defaults", {}).get("max_frames", 81))

        with open(raw_ctrl, "wb") as f:
            f.write(_b64decode(it["control"]["video"]))
        n_frames = normalize_control(raw_ctrl, os.path.join(in_dir, ctrl_name),
                                     it["width"], it["height"], it["fps"], max_frames)

        if it["refs"] and it["ref_mode"] != "none":
            sheet = build_ref_sheet([_b64decode(r["image"]) for r in it["refs"]],
                                    it["width"], it["height"], it["ref_mode"])
            ref_name = tag + "_ref.png"
            sheet.save(os.path.join(in_dir, ref_name))
            tmp.append(os.path.join(in_dir, ref_name))

        wf, info = build_workflow(it, ctrl_name, ref_name, n_frames)
        wait_comfy()
        outputs, gen_s = queue_and_wait(wf, tag)
        out_path = find_video(outputs)
        tmp += [out_path, out_path[:-4] + "_small.mp4"]
        video, out_bytes = deliver(out_path, tag)
        stats = {
            "model": it["model"], "preset": info["preset"], "control": it["control"]["type"],
            "width": it["width"], "height": it["height"], "fps": it["fps"],
            "frames": info["length"], "duration_s": round(info["length"] / it["fps"], 2),
            "control_frames": n_frames, "refs": len(it["refs"]) if ref_name else 0,
            "seed": it["seed"], "comfy_s": round(gen_s, 1),
            "total_s": round(time.time() - t_start, 1), "bytes": out_bytes,
        }
        stats.update(gpu_info())
        return {"video": video, "stats": stats}
    finally:
        for p in tmp:
            try:
                os.remove(p)
            except OSError:
                pass


def handler(job):
    inp = job.get("input") or {}
    job_id = str(job.get("id", "local"))[:24]
    batch = isinstance(inp.get("items"), list)
    items = inp["items"] if batch else [inp]
    results = []
    for i, item in enumerate(items):
        res = {"id": item.get("id")} if batch else {}
        try:
            res.update(process(item, f"{job_id}_{i}"))
        except InputError as e:
            res["error"] = f"entree invalide : {e}"
        except Exception as e:
            traceback.print_exc()
            res["error"] = f"{type(e).__name__}: {e}"
        results.append(res)
    return {"results": results} if batch else results[0]


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
