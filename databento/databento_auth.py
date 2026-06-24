import subprocess
from pathlib import Path


DEFAULT_API_KEY_FILE = Path("secrets/databento_api_key.gpg")


def read_databento_api_key(path: Path | str = DEFAULT_API_KEY_FILE) -> str:
    key_path = Path(path)
    if not key_path.exists():
        raise SystemExit(
            f"Databento API key file not found: {key_path}. "
            "Create it with gpg or pass --api-key-file."
        )

    try:
        result = subprocess.run(
            ["gpg", "--quiet", "--decrypt", str(key_path)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("gpg is not installed or is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Failed to decrypt Databento API key file: {key_path}"
        ) from exc

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SystemExit(
            f"Expected exactly one non-empty API key line in decrypted file: {key_path}"
        )
    return lines[0]
