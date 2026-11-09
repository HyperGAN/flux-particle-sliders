"""Read bounded metadata and refuse unqualified native Flux GPU execution.

No network access, model construction, weight reads, CUDA calls, hooks, or
training execution are provided. Cache presence can never enable readiness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

PRODUCT_BASE = "8a80ace99ad5afa3f2b3d779b14cf55ed557ceb1"
CORE_COMMIT = "a794e3b4a8813b012f2c4faa3da82b78d4a90214"
MAX_METADATA_BYTES = 1024 * 1024
COMPONENTS = ("transformer", "text_encoder", "text_encoder_2", "vae")
METADATA = ("model_index.json", "transformer/config.json", "text_encoder/config.json",
            "text_encoder_2/config.json", "vae/config.json", "scheduler/scheduler_config.json")


def metadata_json(path):
    size = path.stat().st_size
    if size > MAX_METADATA_BYTES:
        raise ValueError("Metadata exceeds the one MiB inspection cap.")
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Metadata grew beyond the one MiB inspection cap.")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON metadata object.")
    return value, dict(size_bytes=size, sha256=hashlib.sha256(raw).hexdigest())


def adapter_estimate(config):
    keys = ("num_layers", "num_single_layers", "num_attention_heads", "attention_head_dim")
    if any(type(config.get(k)) is not int or config[k] < 1 for k in keys):
        return None
    width = config["num_attention_heads"] * config["attention_head_dim"]
    leaves = 4 * config["num_layers"] + 3 * config["num_single_layers"]
    parameters = leaves * (width * 4 + 4 * width)
    return dict(estimate_only=True, rank=4, hidden_width=width, linear_leaves=leaves,
                adapter_parameters=parameters, fp32_adapter_weight_bytes=4 * parameters,
                fp32_weights_gradients_two_adam_moments_bytes=16 * parameters,
                basis="Assumes the unchanged full Attention target policy and square native projections; actual loaded target shapes still require verification.",
                excluded="Frozen base, text encoders, VAE, critic, activations, attention workspaces, optimizer overhead, and CUDA context.")


def inspect_snapshot(snapshot, *, model_id=None, revision=None):
    result = dict(model_id=model_id, requested_revision=revision, metadata={},
                  components={}, missing=[], metadata_files_present=False,
                  weight_integrity="Not verified. Weight payloads are never read or hashed.")
    if snapshot is None:
        result["missing"].append("No candidate snapshot path was supplied.")
        return result
    snapshot = Path(snapshot)
    result["snapshot_path"] = str(snapshot)
    result["snapshot_directory_revision"] = snapshot.name
    if not snapshot.is_dir():
        result["missing"].append("The candidate snapshot directory is absent.")
        return result
    if not model_id or not revision:
        result["missing"].append("An explicit model ID and immutable revision are missing.")
    if revision and (not re.fullmatch(r"[0-9a-f]{40}", revision) or snapshot.name != revision):
        result["missing"].append("The declared immutable revision does not match the snapshot directory.")
    documents = {}
    for relative in METADATA:
        path = snapshot / relative
        try:
            documents[relative], result["metadata"][relative] = metadata_json(path)
        except (OSError, ValueError) as exc:
            result["missing"].append(f"Missing or invalid {relative}: {exc}")
    config = documents.get("transformer/config.json", {})
    result["pipeline_class"] = documents.get("model_index.json", {}).get("_class_name")
    if result["pipeline_class"] != "FluxPipeline":
        result["missing"].append("The pipeline metadata does not declare FluxPipeline.")
    result["transformer_contract"] = {k: config.get(k) for k in (
        "_class_name", "_diffusers_version", "in_channels", "out_channels", "patch_size",
        "num_layers", "num_single_layers", "num_attention_heads", "attention_head_dim",
        "joint_attention_dim", "pooled_projection_dim", "guidance_embeds", "torch_dtype")}
    if config.get("_class_name") != "FluxTransformer2DModel" or config.get("in_channels") != 64:
        result["missing"].append("The metadata does not establish the native packed-64 Flux transformer contract.")
    for component in COMPONENTS:
        folder = snapshot / component
        weights = sorted(folder.glob("*.safetensors")) if folder.is_dir() else []
        entries = []
        for path in weights:
            try:
                size = path.stat().st_size
                entries.append(dict(name=path.name, size_bytes=size))
                if size < 1:
                    result["missing"].append(f"Empty weight file metadata in {component}/{path.name}.")
            except OSError:
                result["missing"].append(f"Broken or missing weight link in {component}/{path.name}.")
        if not entries:
            result["missing"].append(f"No weight file metadata is present for {component}.")
        indexes = sorted(folder.glob("*.index.json")) if folder.is_dir() else []
        if any("-of-" in path.name for path in weights) and not indexes:
            result["missing"].append(f"Sharded weights in {component} have no shard index.")
        for index in indexes:
            try:
                document, identity = metadata_json(index)
                mapping = document.get("weight_map")
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError("The shard weight map is missing.")
                for shard in set(mapping.values()):
                    if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
                        raise ValueError("A shard reference is not a local safetensors basename.")
                    target = folder / shard
                    if not target.is_file() or target.stat().st_size < 1:
                        result["missing"].append(f"Missing indexed shard: {component}/{shard}.")
                result["metadata"][str(index.relative_to(snapshot))] = identity
            except (OSError, ValueError, TypeError) as exc:
                result["missing"].append(f"Invalid shard index {component}/{index.name}: {exc}")
        result["components"][component] = dict(weight_files=entries,
                                                stored_weight_bytes=sum(v["size_bytes"] for v in entries))
    for component, assets in (("tokenizer", ("tokenizer_config.json", "vocab.json", "merges.txt")),
                              ("tokenizer_2", ("tokenizer_config.json",))):
        for asset in assets:
            path = snapshot / component / asset
            if not path.is_file() or path.stat().st_size < 1:
                result["missing"].append(f"Missing tokenizer asset: {component}/{asset}.")
    if not any((snapshot / "tokenizer_2" / name).is_file() and
               (snapshot / "tokenizer_2" / name).stat().st_size > 0
               for name in ("tokenizer.json", "spiece.model")):
        result["missing"].append("The second tokenizer has neither tokenizer.json nor spiece.model.")
    result["adapter_resource_estimate"] = adapter_estimate(config)
    result["metadata_files_present"] = not result["missing"]
    return result


def probe_admission(image_tokens):
    try:
        from importlib.metadata import distribution
        from ps_flux_packed_preview.native import PackedTrainingUnavailable, preflight_training
        from ps_flux_packed_preview.provenance import CORE_COMMIT as installed_core
        origin = json.loads(distribution("particle-sliders-core").read_text("direct_url.json"))
        if installed_core != CORE_COMMIT or origin.get("vcs_info", {}).get("commit_id") != CORE_COMMIT:
            return dict(status="source_mismatch", message="The installed product/core origin does not match the reviewed baseline.")
        try:
            preflight_training((1, image_tokens, 64))
        except PackedTrainingUnavailable as exc:
            return dict(status="blocked", message="The reviewed installed core rejects native packed rank-3 responses before model loading.",
                        product_error=str(exc), cause=str(exc.__cause__),
                        core_commit=installed_core, model_loaded=False)
        return dict(status="unexpected_admission", message="Stop for source review; no native GPU or lifecycle support is established.")
    except Exception as exc:
        return dict(status="source_or_dependency_unavailable", message=f"{type(exc).__name__}: {exc}")


def readiness_report(*, snapshot=None, model_id=None, revision=None, image_tokens, text_tokens,
                     admission_probe=None):
    if any(type(v) is not int or v < 1 for v in (image_tokens, text_tokens)):
        raise ValueError("Positive, explicitly declared image and text token counts are required.")
    admission_probe = admission_probe or probe_admission
    try:
        cache = inspect_snapshot(snapshot, model_id=model_id, revision=revision)
    except OSError as exc:
        cache = dict(metadata_files_present=False,
                     missing=[f"Cache metadata inspection failed: {type(exc).__name__}: {exc}"])
    return dict(schema="flux-gpu-readiness-metadata-v1", ready=False, exit_code=2,
                product_base_head=PRODUCT_BASE, core_commit=CORE_COMMIT,
                requested_response_shape=[1, image_tokens, 64], requested_text_tokens=text_tokens,
                admission=admission_probe(image_tokens), cache=cache,
                blockers=[
                    "The shared continuous_field and response capability contracts do not admit the native packed rank-3 response.",
                    "FluxLoRA is qualified and implemented for frozen float32 CPU models only; CUDA and mixed precision are refused.",
                    "No product-owned immutable pretrained model loader or frozen response dataset/provider is bound.",
                    "No native packed critic, shared training backend, checkpoint/resume, EMA, or shared export API is implemented.",
                ],
                execution="This command performs metadata/source checks only. It cannot launch training, allocate a model, or enable readiness.",
                memory_measurement="None. File sizes are storage metadata, and adapter figures are conditional estimates only.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--image-tokens", type=int, required=True)
    parser.add_argument("--text-tokens", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        report = readiness_report(snapshot=args.snapshot, model_id=args.model_id, revision=args.revision,
                                  image_tokens=args.image_tokens, text_tokens=args.text_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
