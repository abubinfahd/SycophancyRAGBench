"""
Runtime environment management: Google Colab, Kaggle Notebooks, and Local/Server.
Handles storage mounting, output directory bootstrapping, Hugging Face auth, and packaging results into zip.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

# Add package root to sys.path
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from configs import config as cfg


def is_kaggle() -> bool:
    """Return True if executing inside Kaggle Notebooks."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/working")


def is_colab() -> bool:
    """Return True if executing inside Google Colab."""
    if is_kaggle():
        return False
    return "google.colab" in sys.modules or os.path.exists("/content")


def get_runtime_name() -> str:
    """Human-readable string of the current runtime."""
    if is_colab():
        return "Google Colab"
    if is_kaggle():
        return "Kaggle Notebooks"
    return "Local / Self-hosted Server"


def get_hf_token(interactive_prompt: bool = True) -> str:
    """
    Retrieve Hugging Face token across platforms:
    - Colab: google.colab.userdata ('HF_TOKEN')
    - Kaggle: kaggle_secrets UserSecretsClient ('HF_TOKEN')
    - Environment: os.environ.get('HF_TOKEN')
    - Interactive prompt: getpass / input prompt if not configured
    """
    # 1. Environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()

    # 2. Google Colab secrets
    if is_colab():
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if token:
                return token.strip()
        except Exception:
            pass

    # 3. Kaggle secrets
    if is_kaggle():
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            for k_name in ["HF_TOKEN", "hf_token", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF", "huggingface"]:
                try:
                    token = user_secrets.get_secret(k_name)
                    if token:
                        print(f"[Auth] Successfully loaded Hugging Face token from Kaggle Secrets ('{k_name}')!")
                        return token.strip()
                except Exception:
                    pass
        except Exception as e:
            print(f"[Auth] Note: Kaggle Secrets could not be read ({e}).")

    # 4. Interactive prompt (works in Colab, Kaggle, and Terminal)
    if interactive_prompt:
        try:
            import getpass
            print("\n" + "=" * 60)
            print("🔑 Hugging Face Authentication Token Needed for Gated Models (Llama-3.2, Gemma-3)")
            print("Get your free token at: https://huggingface.co/settings/tokens")
            entered = getpass.getpass("Paste your Hugging Face Token (or press Enter to skip): ").strip()
            print("=" * 60 + "\n")
            if entered:
                return entered
        except Exception:
            try:
                entered = input("Paste your Hugging Face Token (or press Enter to skip): ").strip()
                if entered:
                    return entered
            except Exception:
                pass

    return ""


def authenticate_hf(token: str = None, interactive: bool = True) -> bool:
    """Authenticate with Hugging Face Hub if token is available."""
    tok = token or get_hf_token(interactive_prompt=interactive)
    if not tok:
        print("[Auth] No Hugging Face token provided.")
        print("       Non-gated models (Qwen2.5) will download normally.")
        print("       Gated models (Gemma-3, Llama-3.2) require an HF_TOKEN.")
        return False
    try:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)
        print("[Auth] Successfully authenticated with Hugging Face Hub!")
        return True
    except Exception as e:
        print(f"[Auth] Warning: Hugging Face authentication failed: {e}")
        return False


def is_drive_mounted() -> bool:
    """Return True only if /content/drive is genuinely mounted to Google Drive (not a local folder)."""
    if not os.path.exists("/content/drive"):
        return False
    if os.path.ismount("/content/drive"):
        return True
    # Also check if /content/drive/MyDrive is a mount or symlink created by Colab drive.mount
    if os.path.ismount("/content/drive/MyDrive") or os.path.islink("/content/drive/MyDrive"):
        return True
    try:
        with open("/proc/mounts", "r") as f:
            mounts = f.read()
        if "/content/drive" in mounts or "google-drive-ocamlfuse" in mounts or "drivefs" in mounts:
            return True
    except Exception:
        pass
    return False


def setup_workspace(mount_google_drive: bool = True) -> str:
    """
    Setup workspace directories and initialize baseline data.
    - If Colab: verifies real Google Drive mount (recovering any unmounted local /content/drive folder) and ensures directories exist.
    - If Kaggle: prepares /kaggle/working/SycophancyRAGBench.
    - If Local: prepares ./artifacts.
    - Copies pre-bundled benchmark data if not yet present in output directory.
    """
    runtime = get_runtime_name()
    print(f"[Workspace] Initializing environment for: {runtime}")

    if is_colab() and mount_google_drive:
        backup_dir = "/content/_unmounted_drive_backup"
        try:
            from google.colab import drive
            if not is_drive_mounted():
                # If /content/drive was accidentally created as a local folder, move it out of the way so drive.mount succeeds
                if os.path.exists("/content/drive") and os.listdir("/content/drive"):
                    print("[Workspace] Found unmounted local /content/drive directory! Moving to temporary backup before mounting real Google Drive...")
                    if os.path.exists(backup_dir):
                        shutil.rmtree(backup_dir)
                    shutil.move("/content/drive", backup_dir)
                    os.makedirs("/content/drive", exist_ok=True)
                print("[Workspace] Mounting Google Drive to /content/drive ...")
                drive.mount("/content/drive")

            if is_drive_mounted() and os.path.exists("/content/drive/MyDrive"):
                cfg.set_project_dir("/content/drive/MyDrive/SycophancyRAGBench")
                print(f"[Workspace] Google Drive is genuinely mounted! Saving live to: {cfg.PROJECT_DIR}")
                # If we moved an unmounted local backup earlier, copy any generated files into real Google Drive
                unmounted_proj = os.path.join(backup_dir, "MyDrive", "SycophancyRAGBench")
                if os.path.exists(unmounted_proj):
                    print("[Workspace] Restoring files from unmounted local folder into your real Google Drive...")
                    shutil.copytree(unmounted_proj, cfg.PROJECT_DIR, dirs_exist_ok=True)
            else:
                raise RuntimeError("/content/drive/MyDrive is not mounted after drive.mount()")
        except Exception as e:
            print(f"[Workspace] WARNING: Google Drive is NOT mounted ({e}).")
            print("[Workspace] Falling back to local Colab disk: /content/SycophancyRAGBench")
            print("[Workspace] TIP: To save to Google Drive, run `from google.colab import drive; drive.mount('/content/drive')` in a notebook cell first!")
            cfg.set_project_dir("/content/SycophancyRAGBench")

    # Create all required project subdirectories
    for d in cfg.ALL_DIRS:
        os.makedirs(d, exist_ok=True)

    # Initialize bundled datasets into active workspace if absent
    _sync_bundled_data()

    print(f"[Workspace] Project root is: {cfg.PROJECT_DIR}")
    return cfg.PROJECT_DIR


def _sync_bundled_data():
    """Copy bundled claims, prompts, gold passages, distractors, and precomputed artifacts to working directory if not present."""
    pairs = [
        (cfg.BUNDLED_CLAIMS, cfg.CLAIMS_PATH),
        (cfg.BUNDLED_PROMPTS, cfg.PROMPTS_PATH),
        (cfg.BUNDLED_GOLD, cfg.GOLD_PASSAGES_PATH),
        (cfg.BUNDLED_DISTRACTORS, cfg.DISTRACTORS_PATH),
        (cfg.BUNDLED_HUMAN_LABELS, cfg.MANUAL_LABELS),
    ]
    for src, dst in pairs:
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            print(f"[Workspace] Initialized: {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")

    # Sync precomputed retrieval, generations, and eval artifacts if bundled.
    # If the destination file already exists on Google Drive (from a previous or interrupted run),
    # merge any missing precomputed rows without losing a single checkpointed record on Drive.
    precomp_dir = getattr(cfg, "BUNDLED_PRECOMPUTED_DIR", None)
    if precomp_dir and os.path.exists(precomp_dir):
        from src import utils as U
        subdir_map = {
            "retrieval": (cfg.RETRIEVAL_DIR, "claim_id"),
            "generations": (cfg.GEN_DIR, "uid"),
            "eval": (cfg.EVAL_DIR, "ekey"),
        }
        for sub_name, (target_dir, key_col) in subdir_map.items():
            src_sub = os.path.join(precomp_dir, sub_name)
            if os.path.exists(src_sub):
                os.makedirs(target_dir, exist_ok=True)
                for fname in os.listdir(src_sub):
                    s_file = os.path.join(src_sub, fname)
                    d_file = os.path.join(target_dir, fname)
                    if not os.path.isfile(s_file):
                        continue
                    if not os.path.exists(d_file):
                        shutil.copy(s_file, d_file)
                        print(f"[Workspace] Seeded precomputed {sub_name}/{fname} ({os.path.getsize(d_file):,} bytes)")
                    elif fname.endswith(".jsonl"):
                        existing_keys = U.done_keys(d_file, key_col)
                        src_rows = U.read_jsonl(s_file)
                        missing_rows = [r for r in src_rows if str(r.get(key_col, "")) not in existing_keys]
                        if missing_rows:
                            with U.JsonlAppender(d_file) as app:
                                for r in missing_rows:
                                    app.write(r)
                            print(f"[Workspace] Merged {len(missing_rows)} missing precomputed rows into Drive checkpoint {sub_name}/{fname}")


    # Discover and merge any user-saved checkpoint files or output zips anywhere inside PROJECT_DIR or MyDrive
    _discover_and_merge_drive_checkpoints()


def _merge_jsonl_into(src_file: str, dst_file: str, key_col: str, label: str = ""):
    """Safely merge records from src_file into dst_file by key_col without duplicating or losing existing rows."""
    from src import utils as U
    if not os.path.isfile(src_file):
        return
    if os.path.exists(dst_file) and os.path.abspath(src_file) == os.path.abspath(dst_file):
        return
    if not os.path.exists(dst_file):
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)
        print(f"[Checkpoint Recovery] Imported {label or os.path.basename(dst_file)} ({os.path.getsize(dst_file):,} bytes)")
        return
    existing_keys = U.done_keys(dst_file, key_col)
    src_rows = U.read_jsonl(src_file)
    missing_rows = [r for r in src_rows if str(r.get(key_col, "")) not in existing_keys]
    if missing_rows:
        with U.JsonlAppender(dst_file) as app:
            for r in missing_rows:
                app.write(r)
        print(f"[Checkpoint Recovery] Merged {len(missing_rows)} rows from {src_file} -> {os.path.basename(dst_file)}")


def _discover_and_merge_drive_checkpoints():
    """
    Scan PROJECT_DIR (My Drive/SycophancyRAGBench) and MyDrive root for any saved checkpoint .jsonl files
    or output zip archives (e.g. placed in root, subfolders, or sycophancy_rag_bench_outputs.zip)
    and merge them into the active generations/, retrieval/, and eval/ directories.
    """
    search_roots = [cfg.PROJECT_DIR]
    if is_colab() and os.path.exists("/content/drive/MyDrive"):
        # Also check common backup folders on MyDrive
        for extra in [
            "/content/drive/MyDrive/SycophancyRAGBench_artifacts",
            "/content/drive/MyDrive/SycophancyRAGBench_checkpoints",
            "/content/SycophancyRAGBench"
        ]:
            if os.path.exists(extra) and os.path.abspath(extra) not in [os.path.abspath(r) for r in search_roots]:
                search_roots.append(extra)

    # 1. If there is a saved zip on Drive (sycophancy_rag_bench.zip or sycophancy_rag_bench_outputs.zip),
    # extract any checkpoints inside it so we never lose previous progress
    zip_backups = [
        "/content/drive/MyDrive/sycophancy_rag_bench.zip",
        "/content/drive/MyDrive/sycophancy_rag_bench_outputs.zip",
        os.path.join(cfg.PROJECT_DIR, "sycophancy_rag_bench.zip"),
        os.path.join(cfg.PROJECT_DIR, "sycophancy_rag_bench_outputs.zip"),
    ]
    for zb in zip_backups:
        if os.path.isfile(zb):
            tmp_extract = os.path.join("/content" if is_colab() else cfg.PROJECT_DIR, "_recovered_from_drive_zip")
            try:
                with zipfile.ZipFile(zb, "r") as zf:
                    for member in zf.namelist():
                        if member.endswith(".jsonl") and any(k in member for k in ("generations/", "eval/", "retrieval/", "precomputed/")):
                            zf.extract(member, tmp_extract)
                if os.path.exists(tmp_extract) and tmp_extract not in search_roots:
                    search_roots.append(tmp_extract)
            except Exception:
                pass

    # 2. Walk search roots for any .jsonl checkpoint files
    for s_root in search_roots:
        if not os.path.exists(s_root):
            continue
        for root, dirs, files in os.walk(s_root):
            if "index" in root or "collection" in root:
                continue
            for fname in files:
                if not fname.endswith(".jsonl"):
                    continue
                full_p = os.path.join(root, fname)
                if fname == "judged.jsonl":
                    _merge_jsonl_into(full_p, cfg.JUDGED_PATH, "ekey", label="eval/judged.jsonl")
                elif fname.startswith("retrieval_"):
                    dst_p = os.path.join(cfg.RETRIEVAL_DIR, fname)
                    _merge_jsonl_into(full_p, dst_p, "claim_id", label=f"retrieval/{fname}")
                elif "__" in fname and fname not in ("claims.jsonl", "prompts.jsonl", "gold.jsonl", "distractors.jsonl"):
                    dst_p = os.path.join(cfg.GEN_DIR, fname)
                    _merge_jsonl_into(full_p, dst_p, "uid", label=f"generations/{fname}")


def reset_workspace_cache(clear_data: bool = False):
    """
    Protected cache reset: prevents accidental deletion of Google Drive checkpoints
    even if an older Colab notebook cell passes FRESH_RUN = True.
    To genuinely wipe checkpoints, set environment variable FORCE_WIPE_CHECKPOINTS=1.
    """
    if os.environ.get("FORCE_WIPE_CHECKPOINTS") != "1":
        print("\n[Checkpoint Protection] Shield active: Ignoring FRESH_RUN=True so your saved Google Drive checkpoints are NEVER deleted!")
        print("[Checkpoint Protection] Resuming directly from existing checkpoints in:", cfg.PROJECT_DIR)
        _sync_bundled_data()
        return

    print("\n[Fresh Run] FORCE_WIPE_CHECKPOINTS=1 detected. Purging previous caches, model generations, and retrieval indexes...")
    dirs_to_clean = [
        cfg.GEN_DIR, cfg.EVAL_DIR, cfg.RETRIEVAL_DIR,
        cfg.STATS_DIR, cfg.FIG_DIR, cfg.TABLE_DIR, cfg.RELEASE_DIR
    ]

    for d in dirs_to_clean:
        if os.path.exists(d):
            for item in os.listdir(d):
                p = os.path.join(d, item)
                try:
                    if os.path.isfile(p) or os.path.islink(p):
                        os.unlink(p)
                    elif os.path.isdir(p):
                        shutil.rmtree(p)
                except Exception as e:
                    print(f"[Fresh Run] Warning removing {p}: {e}")

    # Remove dense embedding cache
    dense_cache = os.path.join(cfg.CORPUS_DIR, "dense_embeddings.pt")
    if os.path.exists(dense_cache):
        try:
            os.remove(dense_cache)
            print("[Fresh Run] Removed cached dense embeddings.")
        except Exception:
            pass

    # Optional: also clear claims and prompts to force rebuild from raw sources
    if clear_data:
        for p in [cfg.CLAIMS_PATH, cfg.PROMPTS_PATH]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    # Re-ensure all clean subdirectories exist
    for d in cfg.ALL_DIRS:
        os.makedirs(d, exist_ok=True)

    print("[Fresh Run] Workspace completely purged. Fresh run guaranteed with ZERO cache!\n")


def package_outputs(zip_path: str = None, exclude_large_index: bool = True) -> str:
    """
    Compress all generated results, figures, tables, metrics, and models
    into a single zip file for download or archival.
    Essential for Kaggle notebook outputs!
    """
    target_zip = zip_path or cfg.OUTPUT_ZIP_PATH
    os.makedirs(os.path.dirname(target_zip), exist_ok=True)

    print(f"[Packaging] Bundling outputs from {cfg.PROJECT_DIR} into {target_zip} ...")
    count = 0
    total_bytes = 0

    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(cfg.PROJECT_DIR):
            if exclude_large_index and "index" in root:
                continue  # skip large Lucene index files to keep zip lightweight
            for f in files:
                if f.endswith(".zip"):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, cfg.PROJECT_DIR)
                zf.write(full_path, arcname=rel_path)
                count += 1
                total_bytes += os.path.getsize(full_path)

    zip_size_mb = os.path.getsize(target_zip) / (1024 * 1024)
    print(f"[Packaging] Packed {count} files ({total_bytes / (1024*1024):.1f} MB uncompressed) "
          f"-> {target_zip} ({zip_size_mb:.2f} MB)")

    # On Google Colab, also save a copy of the output ZIP directly inside Google Drive
    if is_colab() and os.path.exists("/content/drive/MyDrive"):
        drive_zip = os.path.join(cfg.PROJECT_DIR, "sycophancy_rag_bench_outputs.zip")
        if os.path.abspath(target_zip) != os.path.abspath(drive_zip):
            try:
                shutil.copy2(target_zip, drive_zip)
                print(f"[Packaging] Saved persistent copy to Google Drive -> {drive_zip}")
            except Exception as e:
                print(f"[Packaging] Note: Could not copy zip to Drive ({e})")

    return target_zip
