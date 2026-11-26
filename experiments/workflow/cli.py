"""Short primary-entry commands for the registered paper experiments."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

CELLS = ('R0F0', 'R0F1', 'R0F2', 'R0F3', 'R0F4', 'R1F0')
VARIANTS = ('FULL', 'CONTENT_ZERO', 'FACT_QUALITY_CONSTANT', 'NO_D1', 'NO_SIDE', 'NO_PATH')
METHODS = ('FULL', 'HGT_D1_MATCHED', 'PAIR_MLP_D1_MATCHED', 'HGTDR_ADAPTED',
           'HGTDR_ADAPTED_CONTRA', 'TXGNN_ADAPTED', 'TRAIN_VISIBLE_POPULARITY')
CASES = ('cold_drug', 'cold_disease', 'double_cold', 'first_approval_year_cohort')


def parser():
    p = argparse.ArgumentParser(description='EviHGT 论文实验：自动准备、串行训练和结果汇总')
    p.add_argument('--experiment', required=True, choices=('ablations', 'losses', 'comparisons', 'cold', 'all'))
    p.add_argument('--cells', nargs='+', choices=CELLS, help='选择开发折；省略时使用该实验的论文范围')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, help='ablations 的变体')
    p.add_argument('--methods', nargs='+', choices=METHODS, help='comparisons 的方法')
    p.add_argument('--cases', nargs='+', choices=CASES, help='cold 的场景')
    p.add_argument('--skip-cold-hgtdr', action='store_true', help='冷启动只运行 EviHGT')
    p.add_argument('--mode', choices=('train', 'prepare'), default='train')
    p.add_argument('--config', help='训练 YAML；默认 configs/default.yaml')
    p.add_argument('--raw-config', help='原始数据库 YAML；默认 configs/raw.yaml')
    p.add_argument('--run-dir', help='固定结果目录；相同命令核验后复用完成的任务')
    p.add_argument('--resume', action='store_true', help='从相同 run-dir 内未完成任务的最近检查点恢复')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--smoke', action='store_true', help='缩小折数和训练步数的合成 CPU 软件测试')
    p.add_argument('--txgnn-python', help='TxGNN 独立 CPython 3.11 解释器')
    p.add_argument('--txgnn-source', help='已应用兼容补丁的固定版本 TxGNN 源目录')
    return p


def build_jobs(args):
    jobs = {}
    groups = ('ablations', 'losses', 'comparisons', 'cold') if args.experiment == 'all' else (args.experiment,)
    def add(kind, group, cell, method, loss='softplus', protocol='standard'):
        key = f'{kind}_{protocol}_{cell}_{method}_{loss}'
        if key in jobs:
            if group not in jobs[key]['groups']:
                jobs[key]['groups'].append(group)
        else:
            jobs[key] = dict(id=key, kind=kind, group=group, groups=[group], cell=cell,
                             method=method, loss=loss, protocol=protocol)
    for group in groups:
        defaults = dict(ablations=('R0F1', 'R1F0'), losses=('R0F0', 'R0F1', 'R1F0'), comparisons=CELLS[:5])
        cells = args.cells or (defaults[group][:1] if args.smoke else defaults[group]) if group != 'cold' else ()
        if group == 'ablations':
            for cell in cells:
                for variant in args.variants or VARIANTS[:3]:
                    add('ablation', group, cell, variant)
        elif group == 'losses':
            for cell in cells:
                if cell not in defaults[group]:
                    raise ValueError('losses 支持原配方 R0F0、R0F1、R1F0')
                for loss in ('softplus', 'sigmoid'):
                    add('ablation', group, cell, 'FULL', loss, 'losses' if cell == 'R0F0' else 'standard')
        elif group == 'comparisons':
            for cell in cells:
                for method in args.methods or METHODS:
                    add('full' if method == 'FULL' else 'comparison', group, cell, method)
        else:
            for case in args.cases or (('cold_disease',) if args.smoke else CASES):
                add('cold', group, case, 'FULL')
                if case == 'cold_disease' and not args.skip_cold_hgtdr:
                    add('cold_hgtdr', group, case, 'HGTDR')
    return list(jobs.values())


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if callable(getattr(stream, 'reconfigure', None)):
            stream.reconfigure(encoding='utf-8', errors='replace')
    p = parser()
    args = p.parse_args(argv)
    for field, group in (('variants', 'ablations'), ('methods', 'comparisons'), ('cases', 'cold')):
        if getattr(args, field) and args.experiment not in (group, 'all'):
            p.error(f'--{field} 用于 --experiment {group} 或 all')
    if args.cells and args.experiment == 'cold':
        p.error('cold 使用 --cases')
    if args.resume and not args.run_dir:
        p.error('--resume 要求明确的 --run-dir')
    if args.smoke and (args.config or args.raw_config):
        p.error('--smoke 使用隔离的 CPU 合成输入，不接受真实数据配置')
    try:
        from evi_hgt.settings import ROOT, Settings, load_settings
        from .execution import run_suite
        settings = Settings(device='cpu', precision='fp32', checkpoint_every=1) if args.smoke else load_settings(args.config)
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        root = Path(args.run_dir).resolve() if args.run_dir else ROOT / 'outputs/experiments' / (tag + '_' + args.experiment)
        run_suite(args, build_jobs(args), settings, root)
        return 0
    except Exception as exc:
        if not getattr(exc, 'evihgt_reported', False):
            import traceback
            path = Path(__file__).resolve().parents[2] / 'outputs/logs/startup_errors.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as handle:
                traceback.print_exception(exc, file=handle)
            print(f'[Error] 实验初始化失败（{type(exc).__name__}）：{exc}\n完整错误日志：{path}', file=sys.stderr)
        return 1
