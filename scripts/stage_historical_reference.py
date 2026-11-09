"""Stage one immutable public test reference; never a model or umbrella checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((Path(__file__).resolve().parents[1] /
                           "tests/historical-reference.json").read_text())
    if args.output.exists():
        data = args.output.read_bytes()
    else:
        request = urllib.request.Request(reference["raw_url"], headers={"User-Agent": "Flux-source-qualification"})
        with urllib.request.urlopen(request, timeout=40) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise SystemExit("Reference SHA256 mismatch; no write performed")
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if blob != reference["source_git_blob"]:
        raise SystemExit("Reference Git blob mismatch; no write performed")
    if not args.output.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:
            output.write(data)
    print("Verified external test reference:", reference["source_git_blob"], reference["sha256"])


if __name__ == "__main__":
    main()
