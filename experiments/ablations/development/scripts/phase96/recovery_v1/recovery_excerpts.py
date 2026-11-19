"""Selected functions from the historical recovery driver. See SHA256.json for the source binding."""


def boundary_accounting(checkpoint_step,rows,in_flight):
    if type(checkpoint_step) is not int or checkpoint_step<1 or not rows:raise ValueError('Complete checkpoint and journal required')
    if any(type(r['optimizer_step']) is not int for r in rows) or [r['optimizer_step'] for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('Journal gaps/duplicates/order/type differ')
    fields=('cell','arm','loss_id','manifest_sha256')
    ref={k:rows[0][k] for k in fields}
    if any(any(r[k]!=ref[k] for k in fields) for r in rows):raise ValueError('Journal identity differs')
    last=rows[-1]['optimizer_step']
    if checkpoint_step>last:raise ValueError('Checkpoint exceeds confirmed journal')
    n=in_flight['step']
    if type(n) is not int or n!=last+1 or n<=checkpoint_step or in_flight.get('member',{}).get('step')!=n:
        raise ValueError('Unexpected in-flight boundary')
    ident=in_flight['identity']
    if ident.get('mode')!='DEVELOPMENT' or any(ident.get(k)!=v for k,v in ref.items()):raise ValueError('In-flight identity differs')
    return dict(confirmed_physical_updates=len(rows),last_confirmed_step=last,rollback_step=checkpoint_step,
        known_replay_updates=last-checkpoint_step,in_flight_step=n,
        unrecorded_physical_updates_min=0,unrecorded_physical_updates_max=1)


def add_recovery_disclosure(result,recovery_receipt):
    r=copy.deepcopy(result);rec=copy.deepcopy(recovery_receipt)
    current_unknown=sum(bool(t.get('optimizer_boundary_unknown')) for t in r.get('trajectories',{}).values())
    lower=r['physical_optimizer_updates']+rec.get('unrecorded_physical_updates_min',0)
    upper=r['physical_optimizer_updates']+rec.get('unrecorded_physical_updates_max',1)+current_unknown
    r.update(recovery=rec,physical_optimizer_updates_known=r['physical_optimizer_updates'],
        physical_optimizer_updates_lower_bound=lower,physical_optimizer_updates_upper_bound=upper,
        physical_optimizer_updates_exact=None,replayed_updates_known=r['replayed_updates'],current_unresolved_optimizer_boundaries=current_unknown,
        physical_count_definition='Original physical_optimizer_updates is durable journal count / lower bound, not exact physical total.')
    key=rec['target_cell']+'/'+rec['target_arm']
    if key in r.get('trajectories',{}):
        r['trajectories'][key]['historical_optimizer_boundary_unknown']=True
        r['trajectories'][key]['unrecorded_physical_updates_min']=rec.get('unrecorded_physical_updates_min',0)
        r['trajectories'][key]['unrecorded_physical_updates_max']=rec.get('unrecorded_physical_updates_max',1)
    r.setdefault('scientific_limits',[]).append(
        'R1F0/E0突然中断，从完整checkpoint7168恢复；7169–7284的116次已知物理更新重放。历史7285是否已执行未知(0–1次)，精确物理总数UNRESOLVED。保留原CUDA flags，不声称恢复后逐位等价。')
    r['costs_note']=r.get('costs_note','')+' Durable journal costs and clipping denominator include known replay; historical in-flight step7285 cost is UNKNOWN, never invented.'
    return r


def patch_worker(cell,arm,manifest):
    from evi_hgt_dev96 import runner as engine
    original_update,original_journal,original_restore,original_capture=engine.execute_update,engine.journal,engine.restore,engine.capture
    target=(cell,arm)==(TARGET_CELL,TARGET_ARM)
    source_rows=[json.loads(x) for x in (REC/'pre_crash/updates.jsonl').read_text(encoding='utf-8').splitlines()] if target else []
    previous={r['optimizer_step']:r for r in source_rows};cursor=[manifest['checkpoint_step'] if target else 0]
    first=[False];runtime={'restore_verified':False};preparation_start=time.monotonic()
    def checked_restore(state,model,optimizer,ident,metadata):
        from evi_hgt.phase9.production_state import training_state_digest
        from evi_hgt_dev94.diagnostics import state_signature
        expected=training_state_digest(dict(model=state['model'],optimizer=state['optimizer'],rng=state['rng'],modes=state['module_modes'],grads=state['gradients']))
        answer=original_restore(state,model,optimizer,ident,metadata)
        if state_signature(model,optimizer)!=expected:raise ValueError('Complete model/optimizer/gradients/modes/RNG restore differs')
        cursor[0]=state['step'];runtime['restore_verified']=True
        write(REC/'RESTORE_VERIFIED.json',dict(cell=cell,arm=arm,checkpoint_step=state['step'],
            checkpoint_sha256=manifest['checkpoint_sha256'],full_state_restored=True,recovery_manifest_sha256=file_sha(REC_MANIFEST),pid=os.getpid()))
        return answer
    def checked_update(assembler,built,optimizer,queries,record=None,**kwargs):
        from dataclasses import asdict
        n=cursor[0]+1;old=previous.get(n)
        if old:
            expected_sampling=manifest['original_conditional_plan_digest'] if record is not None else read(MANIFEST)['cells'][cell]['schedule_digest']
            if (digest([asdict(q) for q in queries])!=old['training_member_sha256'] or
                (digest(record) if record is not None else None)!=old['record_sha256'] or old['sampling_sha256']!=expected_sampling):
                raise ValueError('Replay member/record/sampling identity differs before optimizer')
        result=original_update(assembler,built,optimizer,queries,record,**kwargs);cursor[0]=n
        if old:
            journal(REC/'replay_comparisons.jsonl',dict(cell=cell,arm=arm,step=n,member_record_sampling_equal=True,
                losses={k:dict(original=old[k],replayed=result[k],absolute_error=abs(result[k]-old[k])) for k in ('loss','support_loss','contra_loss')},
                exact_loss_equal=all(old[k]==result[k] for k in ('loss','support_loss','contra_loss')),seconds=result['seconds']))
        return result
    def recovery_journal(path,row):
        if Path(path).name=='updates.jsonl':
            row=dict(row,recovery_manifest_sha256=file_sha(REC_MANIFEST),
                historical_in_flight_step_reexecuted=target and row['optimizer_step']==manifest['in_flight_step'])
        original_journal(path,row)
        if target and Path(path).name=='updates.jsonl' and not first[0]:
            if not runtime['restore_verified']:raise ValueError('Update without verified complete-state recovery')
            write(REC/'FIRST_RECOVERED_UPDATE.json',dict(**row,checkpoint_step=manifest['checkpoint_step'],
                full_state_restore_verified=True,actual_recovered_optimizer_update=True,historical_boundary_unknown=True,
                recovery_worker_preparation_seconds=time.monotonic()-preparation_start-row['seconds'],
                log_path=str(Path(path).resolve())))
            first[0]=True
    def recovery_capture(model,optimizer,ident,step,control,metadata):
        validate_recovery()
        control=copy.deepcopy(control);control['recovery_v1']=dict(manifest_sha256=file_sha(REC_MANIFEST),
            historical_boundary_unknown=target,known_replay_updates=manifest['known_replay_updates'] if target else 0)
        return original_capture(model,optimizer,ident,step,control,metadata)
    engine.restore=checked_restore;engine.execute_update=checked_update;engine.journal=recovery_journal;engine.capture=recovery_capture
    return engine


def summarize_recovery(manifest,*,failed,ui):
    from evi_hgt_dev96 import reporting
    original_write,original_markdown=reporting.write,reporting.markdown
    def final_write(path,value,**kw):
        if Path(path).name=='DEV96_RESULTS.json':
            value=add_recovery_disclosure(value,manifest)
            if (REC/'FIRST_RECOVERED_UPDATE.json').exists():value['costs']['recovery_worker_preparation_seconds']=read(REC/'FIRST_RECOVERED_UPDATE.json')['recovery_worker_preparation_seconds']
        return original_write(path,value,**kw)
    def final_markdown(value):
        original_markdown(add_recovery_disclosure(value,manifest))
        with (OUT/'DEV96_RESULTS.md').open('a',encoding='utf-8') as f:
            f.write('\n## 中断恢复与物理更新不确定性\n\n')
            disclosed=add_recovery_disclosure(value,manifest)
            f.write(f"从7168步完整状态恢复；已知重放116步。历史7285步的optimizer执行状态永久UNRESOLVED（额外0–1次）。已知物理更新={value['physical_optimizer_updates']}，总物理范围=[{disclosed['physical_optimizer_updates_lower_bound']}, {disclosed['physical_optimizer_updates_upper_bound']}]。\n\n")
            f.write('所有已有日志和失败现场保留；clipping与资源统计含已知重放，未知边界耗时不补造。原G=6900采样锚点、数据、seed、模型、loss、预算与评价时点均保持。CUDA非逐位确定性限制继续保留。\n')
    reporting.write=final_write;reporting.markdown=final_markdown
    try:
        # Printing is deferred until recovery fields have been written, so no exact-count claim leaks.
        result=reporting.summarize(failed=failed,ui=None)
        result=add_recovery_disclosure(result,manifest)
        if (REC/'FIRST_RECOVERED_UPDATE.json').exists():result['costs']['recovery_worker_preparation_seconds']=read(REC/'FIRST_RECOVERED_UPDATE.json')['recovery_worker_preparation_seconds']
        ui.stage('FINAL STATUS',result['status']);ui._write(f"COMPLETE DEVELOPMENT TRAJECTORIES = {result['complete_development_trajectories']} / 4")
        ui._write(f"ACTUAL LOGICAL UPDATES = {result['logical_optimizer_updates']}；PHYSICAL KNOWN = {result['physical_optimizer_updates_known']}；PHYSICAL TOTAL RANGE = [{result['physical_optimizer_updates_lower_bound']}, {result['physical_optimizer_updates_upper_bound']}]；EXACT = UNRESOLVED")
        for p in result['milestones']:
            if p['metrics']:ui.evaluation_result(p['cell']+'/'+p['arm'],p['step'],p['metrics'])
        for c,stages in result['comparisons'].items():
            for head,values in stages.get('3G',{}).items():ui._write('3G '+c+'/'+head+' E1−E0：'+str({k:values[k] for k in ('delta_auc','delta_ap','ranking_deltas')}))
        ui._write('新增两格描述性摘要：'+str(result['new_two_cell_equal_weight_descriptive']))
        ui._write('实际成本（已知重放计入，未知边界耗时不补造）：'+str(result['costs']))
        ui._write('恢复：known replay116；historical7285 optimizer outcomeUNKNOWN(0–1)；旧中断原因UNRESOLVED')
        for s in ('OLD G3 = HELD','OUTER = UNSCORED','NEW FINAL = SEALED',str(OUT/'DEV96_RESULTS.md'),str(OUT/'DEV96_RESULTS.json')):ui._write(s)
        return result
    finally:reporting.write=original_write;reporting.markdown=original_markdown
