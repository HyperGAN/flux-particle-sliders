"""Product-owned, source-pinned HyperGAN/HNDL builder for the Flux LoRA graph.

The general source-closure checks are adapted from particle-sliders'
``fibo_hndl.verified_builder`` (MIT). No former product network, core-package
payload, ParticleGAN version, training formula or native qualification is implied.
This module imports only stdlib until checked_builder() is called.
"""
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).parent
PROFILE = "flux-native95-product-hndl010-builder-v1"
HYPERGAN_COMMIT = "5f368a3770cf5bdae0448995c9f76e29af8049f7"
HNDL_VERSION = "0.10.0"
LORA_SHA256 = "c09bb012a479eb103dd3e710a8909251c0cb3a852f45a26112f2a9cc972a30e3"
BUILDER_SOURCES_SHA256 = "96aab47343ef6fed938be056845d5f1b3bfdfeacd36da247c10ab907b3bf46c7"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_builder():
    """Return the exact external build_network and its checked Flux identity.

    Admission binds every HNDL Python source, the reviewed HyperGAN builder
    closure, its clean commit provenance and this product's two-line graph.
    It establishes source identity only; it does not construct a network or
    certify installed models, training updates, continuation or rendering.
    """
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError("Flux HNDL requires Python>=3.11,<3.15")
    manifest = ROOT / "builder-sources.json"
    if _sha(manifest) != BUILDER_SOURCES_SHA256:
        raise ValueError("Flux requires the exact reviewed builder source manifest")
    expected = json.loads(manifest.read_text())
    if (expected.get("schema"), expected.get("profile"),
            expected.get("hndl_version"), expected.get("hypergan_commit"),
            expected.get("lora_sha256"), expected.get("source_policy")) != (
            "flux-native95-product-builder-source-closure-v1", PROFILE,
            HNDL_VERSION, HYPERGAN_COMMIT, LORA_SHA256,
            "all-hndl-python-and-declared-hypergan-builder-files-v1"):
        raise ValueError("Flux builder source manifest contradicts the binding")
    if _sha(ROOT / "lora.hndl") != LORA_SHA256:
        raise ValueError("Flux native LoRA graph differs from the original two-line graph")
    try:
        import hndl
        from hypergan.hndl_networks import build_network
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Flux requires explicit hndl==0.10.0 and the pinned HyperGAN train dependency"
        ) from exc
    if version("hndl") != HNDL_VERSION:
        raise ValueError("Flux requires hndl==0.10.0")
    hndl_root = Path(hndl.__file__).parent
    actual = {str(p.relative_to(hndl_root)): _sha(p)
              for p in hndl_root.rglob("*.py")}
    if actual != expected["hndl"]:
        raise ValueError("Flux HNDL Python source closure differs from the reviewed source")
    source = inspect.getsourcefile(build_network)
    if source is None:
        raise ValueError("Flux requires inspectable HyperGAN build_network source")
    hypergan_root = Path(source).parent
    provenance = hypergan_root / "_build_provenance.json"
    if not provenance.is_file() or json.loads(provenance.read_text()) != {
            "schema_version": 1, "commit": HYPERGAN_COMMIT, "dirty": False}:
        raise ValueError("Flux requires exact clean HyperGAN commit build provenance")
    if any(_sha(hypergan_root / name) != digest
           for name, digest in expected["hypergan"].items()):
        raise ValueError("Flux HyperGAN builder source closure differs from the reviewed source")
    identity = dict(
        profile=PROFILE,
        hndl_version=HNDL_VERSION,
        hypergan_commit=HYPERGAN_COMMIT,
        hypergan_build_provenance_sha256=_sha(provenance),
        builder_sources_sha256=_sha(manifest),
        builder_source_sha256=_sha(__file__),
        source_policy=expected["source_policy"],
        flux_lora_sha256=LORA_SHA256,
    )
    return build_network, identity
