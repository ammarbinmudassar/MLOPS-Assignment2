"""Run this yourself in a terminal to enter the token without echoing it."""

from getpass import getpass
import subprocess

from dotenv import set_key

from task2.common import ROOT


def main():
    ignored = subprocess.run(["git", "check-ignore", "--quiet", ".env"], cwd=ROOT).returncode == 0
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", ".env"], cwd=ROOT,
                             capture_output=True).returncode == 0
    if not ignored or tracked:
        raise SystemExit("Stop: .env must be ignored and untracked before saving a token.")
    token = getpass("Paste your DagsHub token (hidden): ").strip()
    if not token or "\n" in token or "\r" in token:
        raise SystemExit("No valid token entered; settings were not changed.")
    path = ROOT / ".env"
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    set_key(path, "MLFLOW_TRACKING_PASSWORD", token)
    path.chmod(0o600)
    print("Token saved to ignored .env. Its value was not displayed.")


if __name__ == "__main__":
    main()

