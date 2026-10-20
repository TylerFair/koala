"""python -m koala.marginalization: isolated prepare, fit, score, or full grid."""
import argparse
import os

from .grid import prepare_grid, run_grid


def _cpu_affinity(args, manifest):
    """Bound CPU worker creation before importing numerical model modules."""
    from pathlib import Path
    import yaml
    from .grid import resolve_path
    if not hasattr(os, 'sched_getaffinity'):
        return
    count = args.cpu_cores
    if count is None:
        spec = yaml.safe_load(Path(manifest).read_text())
        cfg = yaml.safe_load(resolve_path(manifest, spec['variants'][0]['config']).read_text())
        if cfg.get('host_device', 'cpu') != 'cpu' or os.environ.get('JAX_PLATFORMS', 'cpu') != 'cpu':
            return
        count = 4
    if count < 0:
        raise ValueError('--cpu-cores must be nonnegative; zero retains all available cores.')
    available = sorted(os.sched_getaffinity(0))
    if count and len(available) > count:
        os.sched_setaffinity(0, set(available[:count]))
        print(f'Grid CPU execution: {count} cores (override with --cpu-cores).', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("prepare", "run"):
        command = commands.add_parser(action)
        command.add_argument("base_config")
        command.add_argument("--workspace", required=True)
        command.add_argument("--matrix", help="Explicit variant matrix instead of Cartesian axes")
        command.add_argument("--trends", default="linear,quadratic")
        command.add_argument("--laws", default="quadratic,power2")
        command.add_argument("--treatments", default="fixed,uniform")
    fit = commands.add_parser("fit")
    fit.add_argument("manifest")
    for command in (commands.choices['run'], fit):
        command.add_argument('--execution', choices=('sequential', 'batched'), default='sequential',
                             help='Batched supports fixed/uniform quadratic/power-2 LD with linear, quadratic, or sigmoid-step trends.')
    score = commands.add_parser("score")
    score.add_argument("manifest")
    for command in (commands.choices['run'], fit, score):
        command.add_argument('--cpu-cores', type=int, help='CPU worker cores; default 4 for CPU grids, 0 keeps all.')
    for command in (commands.choices["run"], score):
        command.add_argument("--method", choices=("stacking", "pseudo_bma_plus", "hierarchical_map", "hierarchical", "equal"), default="stacking")
        command.add_argument("--pooling", choices=("channel", "global"), default="channel")
        command.add_argument("--stage", choices=("high_resolution", "low_resolution"), default="high_resolution")
        command.add_argument("--output")
        command.add_argument("--draw-batch-size", type=int, default=32)
        command.add_argument("--point-batch-size", type=int, default=256)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--n-out", type=int, default=10000)
        command.add_argument("--n-boot", type=int, default=1000)
        command.add_argument("--strength", type=float, default=1.)
        command.add_argument("--weight-warmup", type=int, default=500)
        command.add_argument("--weight-draws", type=int, default=1000)
        command.add_argument("--weight-chains", type=int, default=4)
        command.add_argument("--allow-unreliable", action="store_true", help="Write explicitly exploratory products despite high Pareto k")
    args = parser.parse_args(argv)
    if args.command in {"prepare", "run"}:
        manifest = prepare_grid(args.base_config, args.workspace, matrix=args.matrix,
            trends=tuple(args.trends.split(",")), laws=tuple(args.laws.split(",")),
            treatments=tuple(args.treatments.split(",")))
        print(f"Grid manifest: {manifest}", flush=True)
    else:
        manifest = args.manifest
    if args.command != 'prepare':
        _cpu_affinity(args, manifest)
    if args.command in {"run", "fit"}:
        if args.execution == 'batched':
            from .batched_fitting import run_batched_grid
            manifest = run_batched_grid(manifest)
            print(f'Batched scoring manifest: {manifest}', flush=True)
        else:
            run_grid(manifest)
    if args.command in {"run", "score"}:
        import jax
        jax.config.update("jax_enable_x64", True)
        from .scoring import score_grid
        result = score_grid(manifest, args.output, stage_kind=args.stage, method=args.method,
            pooling=args.pooling, draw_batch_size=args.draw_batch_size,
            point_batch_size=args.point_batch_size, n_out=args.n_out, seed=args.seed,
            allow_unreliable=args.allow_unreliable, n_boot=args.n_boot, strength=args.strength,
            warmup=args.weight_warmup, draws=args.weight_draws, chains=args.weight_chains)
        print(f"Marginalized spectrum: {result}")


if __name__ == "__main__":
    main()
