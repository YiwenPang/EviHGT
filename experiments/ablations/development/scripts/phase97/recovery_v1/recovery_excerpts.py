"""Selected functions from the historical recovery driver. See SHA256.json for the source binding."""


def run_authorized_remaining(verify_completed,execute_fresh):
    verify_completed(*COMPLETED)
    return [execute_fresh(cell,arm) for cell,arm in REMAINING]


def reporting_facility(report_out):
    """Redirect only five literal output destinations in the old report AST."""
    report_out=Path(report_out).resolve()
    if not report_out.is_relative_to(REC.resolve()):raise PermissionError('Recovery report namespace required')
    from evi_hgt_dev97 import reporting as parent
    source=Path(parent.__file__);tree=ast.parse(source.read_text('utf-8'),filename=str(source))
    class Destinations(ast.NodeTransformer):
        replacements=0
        def visit_BinOp(self,node):
            self.generic_visit(node)
            if (isinstance(node.op,ast.Div) and isinstance(node.left,ast.Name) and node.left.id=='OUT'
                and isinstance(node.right,ast.Constant) and node.right.value in REPORT_NAMES):
                node.left=ast.copy_location(ast.Name(id='REPORT_OUT',ctx=ast.Load()),node.left);self.replacements+=1
            return node
    change=Destinations();tree=change.visit(tree);ast.fix_missing_locations(tree)
    if change.replacements!=9:raise ValueError('Parent reporting destinations changed; re-audit required: '+str(change.replacements))
    name='evi_hgt_dev97._recovery_v1_report'
    spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
    exec(compile(tree,str(source),'exec'),module.__dict__)
    module.REPORT_OUT=report_out;module.recovery_output_replacements=change.replacements
    return module


def verify_completed(cell,arm):
    if (cell,arm)!=COMPLETED:raise PermissionError('Only the complete first ablation is reused')
    m=p.validate_manifest();return verify_receipt(p.read(OUT/cell/arm/'receipt.json'),p.identity(m,cell,arm))


def summarize(failed,ui=None):
    module=reporting_facility(REC)
    value=module.summarize(failed=failed,ui=ui)
    value['control_recovery']=dict(manifest=str(MANIFEST),manifest_sha256=p.file_sha(MANIFEST),
        complete_first_arm_reused_without_new_training_or_evaluation=True,
        original_partial_failure_retained=str(OUT/'DEV97_RESULTS.json'),
        remaining_queue=[list(x) for x in REMAINING],repair='Pinned installed/signed Adobe AcroCEF classification only; parent scientific code/identity unchanged')
    p.write(REC/'DEV97_RESULTS.json',value,immutable=False)
    with (REC/'DEV97_RESULTS.md').open('a',encoding='utf-8') as f:
        f.write('\n## 控制层修复与继续执行\n\n原第一条20541步及全部评价只读复用，无重训、无额外评价forward。原PARTIAL_FAILED报告和故障证据保留于父目录。仅继续固定后三条，科学身份/初始化/采样/损失/评价/预算均不变。完整模型参照仍只读2条。\n')
    return value


def worker(cell,arm):
    if (cell,arm) not in REMAINING:raise PermissionError('Completed arm cannot be executed by recovery worker')
    validate_recovery();no_conflicts();root=OUT/cell/arm
    if any((root/name).exists() for name in ('started.json','updates.jsonl','IN_FLIGHT_UPDATE.json','FIRST_UPDATE.json')) or list((root/'checkpoints').glob('*.COMMIT.json')):
        raise RuntimeError('Fresh continuation refused existing arm state; separate explicit recovery needed')
    engine=facility(cell,arm,resume=False);engine.no_conflicts=no_conflicts
    original_manifest=engine.validate_manifest
    def checked_manifest():validate_recovery();return original_manifest()
    engine.validate_manifest=checked_manifest
    return engine.run_arm(cell,arm,resume=False)
