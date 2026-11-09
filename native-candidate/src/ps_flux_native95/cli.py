"""Explicit local entry points; --help/check never imports Torch or loads models."""
import argparse
import json
from pathlib import Path
from .contracts import BACKEND_NAME, INITIALIZATION_POLICY, PROTOCOL, SOURCE_PROFILE


def main(argv=None):
    parser = argparse.ArgumentParser(description="New native Flux packed RAW/noEMA cohort; no pretrained qualification claimed")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="allocation-free installed package/profile description")
    cache = sub.add_parser("cache", help="explicit frozen full native positive-response preparation")
    for name in ("model-dir", "model-lock-sha256", "pairs", "output"):
        cache.add_argument("--" + name, required=True)
    cache.add_argument("--device", default="cpu", choices=("cpu", "cuda:0", "cuda:1"))
    cache.add_argument("--base-precision", default="bfloat16", choices=("float32", "bfloat16"))
    cache.add_argument("--height", type=int, default=512); cache.add_argument("--width", type=int, default=512)
    cache.add_argument("--text-tokens", type=int, default=256); cache.add_argument("--steps", type=int, default=4)
    cache.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    train = sub.add_parser("train", help="exact common config and new owned output; fixed total update budget")
    train.add_argument("--config", required=True); train.add_argument("--output", required=True)
    train.add_argument("--resume"); train.add_argument("--stop-after", type=int)
    render = sub.add_parser("render-pair", help="RAW-only true native matched OFF/ON PNGs")
    for name in ("artifact", "model-dir", "model-lock-sha256", "output", "prompt"):
        render.add_argument("--" + name, required=True)
    render.add_argument("--device", default="cpu", choices=("cpu", "cuda:0", "cuda:1"))
    render.add_argument("--precision", default="bfloat16", choices=("float32", "bfloat16"))
    render.add_argument("--steps", type=int, default=4); render.add_argument("--seed", type=int, default=7)
    render.add_argument("--strength", type=float, default=1.)
    args = parser.parse_args(argv)
    if args.command == "check":
        print(json.dumps({"backend": BACKEND_NAME, "protocol": PROTOCOL, "source_profile": SOURCE_PROFILE,
            "initialization_policy": INITIALIZATION_POLICY, "selection": "raw", "ema_decay": None,
            "generator_arm": "adapter_gan", "particle_bank": None, "native_pretrained_qualified": False,
            "status": "new implementation Source; actual matching native checks pending"}, indent=2))
    elif args.command == "cache":
        import torch
        from .contracts import geometry
        from .native import NativeRuntime
        from .adapter import FluxLoRA
        from .cache import prepare_cache
        shape = geometry(vars(args))
        runtime, adapter = None, None
        try:
            runtime = NativeRuntime(args.model_dir, model_lock_sha256=args.model_lock_sha256,
                device=args.device, precision=args.base_precision, encoder=True)
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(0)
                adapter = FluxLoRA(runtime.transformer, image_tokens=shape["image_tokens"],
                                   text_tokens=shape["text_tokens"]).to(runtime.device).attach()
            print(prepare_cache(runtime, adapter, json.loads(Path(args.pairs).read_text()), args.output,
                                geometry=shape, seeds=args.seeds, steps=args.steps))
        finally:
            from particle_sliders.restoration import _restore_all
            _restore_all([*([] if adapter is None else [adapter.close]), *([] if runtime is None else [runtime.close])])
    elif args.command == "train":
        from particle_sliders.lifecycle import SessionConfig
        from .lifecycle import open_training
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        config = SessionConfig.from_dict(json.loads(Path(args.config).read_text()))
        with open_training(config, resume=args.resume) as session:
            def saved(metrics):
                session.checkpoint(output / f"FULL-{metrics['step']:06d}.pt")
            session.train(stop_after=args.stop_after, on_step=saved)
            session.export_inference(output / "RAW.pt", selection="raw")
    else:
        from .render import render_pair
        receipt = render_pair(**{key: value for key, value in vars(args).items() if key != "command"})
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
