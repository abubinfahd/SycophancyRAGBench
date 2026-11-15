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


def setup_workspace(mount_google_drive: bool = True) -> str:
    """
    Setup workspace directories and initialize baseline data.
    - If Colab: mounts Google Drive (if requested) and ensures directories exist.
    - If Kaggle: prepares /kaggle/working/SycophancyRAGBench.
    - If Local: prepares ./artifacts.
    - Copies pre-bundled benchmark data if not yet present in output directory.
    """
    runtime = get_runtime_name()
    print(f"[Workspace] Initializing environment for: {runtime}")

    if is_colab() and mount_google_drive:
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                print("[Workspace] Mounting Google Drive to /content/drive ...")
                drive.mount("/content/drive")
            print("[Workspace] Google Drive mounted successfully.")
        except Exception as e:
            print(f"[Workspace] Could not mount Google Drive ({e}). Falling back to /content/SycophancyRAGBench.")
            cfg.PROJECT_DIR = "/content/SycophancyRAGBench"
            # Update paths to fallback
            for d_name in ["DATA_DIR", "CORPUS_DIR", "COLLECTION_DIR", "INDEX_DIR",
                           "RETRIEVAL_DIR", "GEN_DIR", "EVAL_DIR", "STATS_DIR",
                           "FIG_DIR", "TABLE_DIR", "RELEASE_DIR"]:
                sub = getattr(cfg, d_name).replace(cfg.DRIVE_ROOT, cfg.PROJECT_DIR)
                setattr(cfg, d_name, sub)

    # Create all required project subdirectories
    for d in cfg.ALL_DIRS:
        os.makedirs(d, exist_ok=True)

    # Initialize bundled datasets into active workspace if absent
    _sync_bundled_data()

    print(f"[Workspace] Project root is: {cfg.PROJECT_DIR}")
    return cfg.PROJECT_DIR


def _sync_bundled_data():
    """Copy bundled claims, prompts, gold passages, and distractors to working directory if not present."""
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


def reset_workspace_cache(clear_data: bool = False):
    """
    Purge all cached results, generations, evaluations, figures, and retrieval indexes
    to ensure a 100% clean, fresh run with zero cache.
    """
    print("\n[Fresh Run] Purging previous caches, model generations, and retrieval indexes...")
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
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, cfg.PROJECT_DIR)
                zf.write(full_path, arcname=rel_path)
                count += 1
                total_bytes += os.path.getsize(full_path)

    zip_size_mb = os.path.getsize(target_zip) / (1024 * 1024)
    print(f"[Packaging] Packed {count} files ({total_bytes / (1024*1024):.1f} MB uncompressed) "
          f"-> {target_zip} ({zip_size_mb:.2f} MB)")
    return target_zip
