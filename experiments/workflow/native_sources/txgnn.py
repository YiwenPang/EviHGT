"""TxGNN's native two-stage recipe on the common development panels."""

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import torch
from filelock import FileLock

from evi_hgt.data.common import sha256
from evi_hgt.io import digest, write_json
from evi_hgt.runner import restore_rng, rng_state
from evi_hgt.settings import ROOT
from experiments.native_adapters.EviHGT.src.evi_hgt.ranking_protocol.txgnn_protocol import (
    InnerSelection, fixed_recipe,
)
from experiments.workflow.comparisons import (
    load_training_state, model_state, save_training_state, scored_panel, selection_key,
)
from .graph import native_graph, native_lookup, native_unlabeled

TXGNN_COMMIT = "0e0203c3939f7d12da27aaf10e55daaff5424023"
TXGNN_REPO = "https://github.com/mims-harvard/TxGNN.git"
ARCHIVE = ROOT / "experiments/native_adapters/TxGNN"
PARENT_HASHES = {
    "model.py": "541795c037f717e2d42162c93afd74a044909709c575b04b13f64bdd0b9875c4",
    "utils.py": "4a2f61d907f449db313b01b023c24e1ffc92568087a315e976291a551ad8eca0",
    "graphmask/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "graphmask/multiple_inputs_layernorm_linear.py": "cceba87c5f6a441405ec73b8d7c8aa6bbf8728c3053471a32d4d46c9b9f5549b",
    "graphmask/squeezer.py": "d7a07b9c5f377dc7509f6610fb64cf924292244841f02477bdb05b7191becb40",
    "graphmask/sigmoid_penalty.py": "fcec4982beb68f75651d528f6c817ea586d42e2be733ea6d204ffaff1faf8915",
    "data_splits/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "data_splits/datasplit.py": "74c399e1d0b2e618e781ee23f4298b73492fe9a3168fd718d515b3f3d1418762",
    "data_splits/do_obo_parser.py": "ba068254a057919f0f6c9ca1d657b44057b5f9b1305d2b2f122470dc8d41c4be",
}


def _native_source_root(context, *, smoke):
    if context.get("txgnn_source"):
        return Path(context["txgnn_source"]).resolve()
    cache = Path(context.get("native_cache", ROOT / "data/cache/experiments/native"))
    source = cache / ("TxGNN_" + TXGNN_COMMIT)
    if source.is_dir():
        return source
    if context.get("offline") or smoke:
        raise FileNotFoundError("Populate the pinned TxGNN source cache or supply txgnn_source for offline testing")
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(source) + ".lock"):
        if not source.is_dir():
            stage = source.with_name(source.name + "." + uuid4().hex)
            stage.mkdir()
            def git(*arguments):
                subprocess.run(["git", "-c", "core.autocrlf=false", "-C", str(stage), *arguments],
                               check=True)
            git("init", "--quiet")
            git("fetch", "--depth=1", TXGNN_REPO, TXGNN_COMMIT)
            git("checkout", "--detach", "--quiet", TXGNN_COMMIT)
            git("apply", str(ARCHIVE / "utils-compat.patch"))
            _check_native_source(stage)
            os.rename(stage, source)
    return source


def _check_native_source(source):
    # Git uses LF blobs even where a local Windows checkout uses CRLF.
    result = {name: hashlib.sha256((source / "txgnn" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for name in PARENT_HASHES}
    if result != PARENT_HASHES:
        raise ValueError("Pinned TxGNN source identity differs")
    return result


def native_txgnn(context, *, smoke):
    """Import the pinned native model without invoking upstream data downloads."""
    source = _native_source_root(context, smoke=smoke)
    files = _check_native_source(source)
    package_name = "_evihgt_txgnn_" + digest(str(source))[:16]
    name = package_name + ".competition"
    adapter = ARCHIVE / "txgnn/competition.py"
    if name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(source / "txgnn")]
        sys.modules[package_name] = package
        spec = importlib.util.spec_from_file_location(name, adapter)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name], dict(files, competition=sha256(adapter), revision=TXGNN_COMMIT)


def _pair_tensor(lookup, pairs):
    drugs, diseases = lookup["drug"], lookup["disease"]
    pairs = tuple(pairs)
    return torch.tensor([[drugs[drug] for drug, _ in pairs], [diseases[disease] for _, disease in pairs]],
                        dtype=torch.int64)


def _native_data(native, corpus, context, root, *, smoke):
    import dgl
    nodes, relations, arrays, receipt = native_graph(corpus, context, root, smoke=smoke)
    counts = {name: len(ids) for name, ids in nodes.partition_by("human_node_type", as_dict=True).items()}
    counts = {name[0] if isinstance(name, tuple) else name: size for name, size in counts.items()}
    edges = {}
    for row, array in zip(relations, arrays, strict=True):
        key = native.native_etype(row)
        if {key[0], key[2]} == {"drug", "disease"} and key not in native.DDA_ETYPES:
            raise ValueError("Unverified clinical relation in TxGNN graph")
        if key in edges:
            raise ValueError("Duplicate canonical TxGNN relation")
        edges[key] = tuple(torch.tensor(index, dtype=torch.int64) for index in array)
    empty = torch.empty(0, dtype=torch.int64)
    for etype in native.DDA_ETYPES:
        edges.setdefault(etype, (empty, empty))

    lookup = native_lookup(nodes)
    native._seed(corpus.training_seed)
    graph = native.initialize_node_embedding(dgl.heterograph(edges, num_nodes_dict=counts), 100)
    del edges
    support = _pair_tensor(lookup, sorted(corpus.support))
    contra = _pair_tensor(lookup, sorted(corpus.unsuitability))
    reference = native.training_reference_graph(graph, support, contra)
    with_targets = torch.cat([support, contra], dim=1)
    encoder = native.masked_encoder_graph(graph, with_targets)
    return SimpleNamespace(graph=graph, reference_graph=reference, encoder_graph=encoder,
                           positive_graph=native.dda_graph(graph, support, contra), support=support,
                           contra=contra, lookup=lookup, receipt=receipt)


class TrainingU:
    """Fold-bound uniform U stream with the recorded call-based seed derivation."""
    def __init__(self, corpus, lookup, offset):
        self.policy = native_unlabeled(corpus)
        self.lookup = lookup
        self.seed = corpus.training_seed + offset
        self.calls = 0

    def __call__(self, count):
        if not count:
            return _pair_tensor(self.lookup, ())
        material = hashlib.sha256(f"{self.seed}:{self.calls}".encode()).digest()
        pairs = self.policy.sample(count, seed=int.from_bytes(material[:4], "little"))
        self.calls += 1
        return _pair_tensor(self.lookup, pairs)


@contextmanager
def _dgl_input_threads():
    from dgl import utils
    previous = utils.get_num_threads()
    utils.set_num_threads(1)
    try:
        yield
    finally:
        utils.set_num_threads(previous)


def _update(native, model, optimizer, loss):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite TxGNN loss")
    loss.backward()
    native._check_gradients(model)
    optimizer.step()
    return float(loss.detach())


def _score(native, adapter, data, root, head, members):
    previous, rng = adapter.model.training, rng_state()
    try:
        adapter.model.eval()
        with torch.inference_mode():
            pairs = _pair_tensor(data.lookup, ((row["drug_id"], row["disease_id"]) for row in members))
            graph = native.dda_graph(data.graph, pairs, pairs)
            relation = ("drug", "indication" if head == "support" else "contraindication", "disease")
            scores = adapter.score_graph(graph, etypes=[relation])[relation].tolist()
            return scored_panel(root, head, members, scores)
    finally:
        adapter.model.train(previous)
        restore_rng(rng)


def _capture_auxiliary(model):
    predictor = model.pred
    return copy.deepcopy(dict(
        sim=predictor.sim_all_etypes, profile=predictor.diseases_profile_etypes,
        mapping=predictor.diseaseid2id_etypes,
        gates={key: gate.state_dict() for key, gate in predictor.W_gate.items()},
        gate_gradients={key: {name: parameter.grad for name, parameter in gate.named_parameters()}
                        for key, gate in predictor.W_gate.items()},
    ))


def _restore_auxiliary(model, state):
    predictor = model.pred
    predictor.sim_all_etypes = state["sim"]
    predictor.diseases_profile_etypes = state["profile"]
    predictor.diseaseid2id_etypes = state["mapping"]
    if predictor.W_gate.keys() != state["gates"].keys():
        raise ValueError("TxGNN checkpoint prototype gate layout differs")
    for key, gate in predictor.W_gate.items():
        gate.load_state_dict(state["gates"][key], strict=True)
        for name, parameter in gate.named_parameters():
            parameter.grad = state["gate_gradients"][key][name]


def _write_checkpoint(root, binding, model, optimizer, *, stage, cursor, stream, **state):
    save_training_state(root / "checkpoint.pt", dict(binding=binding, stage=stage, cursor=cursor,
        model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(),
        u_calls=stream.calls, auxiliary=_capture_auxiliary(model), **state))


def run_txgnn(corpus, panels, root, console, *, smoke, checkpoint, context):
    """Execute the native DGL pretraining and selection schedule in a CPU worker."""
    import dgl
    if dgl.__version__ != "1.1.2" or torch.cuda.is_initialized():
        raise RuntimeError("TxGNN requires an isolated DGL 1.1.2 CPU worker")
    native, sources = native_txgnn(context, smoke=smoke)
    with _dgl_input_threads():
        data = _native_data(native, corpus, context, root, smoke=smoke)
    torch.set_num_threads(8 if not smoke else 1)
    adapter = native.NativeCompetitionModel(data.encoder_graph, data.reference_graph, dimension=100)
    model = adapter.model
    seed = corpus.training_seed
    fixed = fixed_recipe(seed)
    epochs = 1 if smoke else fixed["finetune_epochs"]
    binding = dict(method="TXGNN_ADAPTED", corpus=corpus.identity, graph=digest(data.receipt),
                   panels=digest(panels), sources=sources, torch=str(torch.__version__),
                   dgl=dgl.__version__, recipe=fixed, recipe_sha256=sha256(__file__), smoke=smoke)
    saved = load_training_state(checkpoint, binding) if checkpoint else None
    if saved is not None and saved["stage"] not in ("pretrain", "finetune"):
        raise ValueError("Unrecognized native TxGNN training stage")
    pretrain_updates = 0
    if saved is None or saved["stage"] == "pretrain":
        native._seed(fixed["pretrain_seed"])
        stream = TrainingU(corpus, data.lookup, 102)
        loader = native.pretraining_loader(data.graph, data.support, data.contra, stream,
                                           batch_size=fixed["pretrain_batch_size"])
        if loader.num_workers or loader.use_prefetch_thread:
            raise ValueError("Expected a synchronous native DGL loader")
        iterator = iter(loader)
        permutation = loader.dataset._indices.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=fixed["pretrain_lr"])
        if saved is not None:
            model.load_state_dict(saved["model"], strict=True)
            _restore_auxiliary(model, saved["auxiliary"])
            optimizer.load_state_dict(saved["optimizer"])
            pretrain_updates = saved["cursor"]
            stream.calls = saved["u_calls"]
            restored = saved["permutation"]
            if restored.shape != permutation.shape or restored.dtype != torch.int64:
                raise ValueError("TxGNN checkpoint pretraining permutation differs")
            if not torch.equal(torch.sort(restored).values, torch.arange(len(permutation))):
                raise ValueError("TxGNN checkpoint pretraining permutation is invalid")
            permutation = restored
            loader.shuffle = False
            loader.dataset._indices = permutation.clone()
            iterator = iter(loader)
            internal = iterator.dataloader_it._dataset_fetcher.dataset_iter
            if type(internal).__name__ != "_TensorizedDatasetIter" or internal.index != 0:
                raise ValueError("Native DGL resume cursor is incompatible")
            internal.index = pretrain_updates * fixed["pretrain_batch_size"]
            restore_rng(saved["rng"])
        if not 0 <= pretrain_updates <= len(loader):
            raise ValueError("Invalid TxGNN pretraining cursor")
        model.train()
        with console.progress(len(loader) - pretrain_updates, "Epoch 1/1 TxGNN 预训练", unit="update") as progress:
            while pretrain_updates < len(loader):
                with _dgl_input_threads():
                    batch = next(iterator)
                optimizer.zero_grad(set_to_none=True)
                loss = native._pretrain_loss(adapter, batch, data.reference_graph)
                value = _update(native, model, optimizer, loss)
                pretrain_updates += 1
                progress.update(loss=value)
                console.event("optimizer_update", stage="pretrain", update=pretrain_updates, loss=value)
                if pretrain_updates % 32 == 0 or pretrain_updates == len(loader):
                    _write_checkpoint(root, binding, model, optimizer, stage="pretrain",
                                      cursor=pretrain_updates, stream=stream, permutation=permutation)
        if pretrain_updates != math.ceil(sum(map(len, native.pretraining_edge_ids(
                data.graph, data.support, data.contra).values())) / fixed["pretrain_batch_size"]):
            raise ValueError("Native TxGNN pretraining did not cover the legal edge domain")
        del iterator, loader
        native._seed(fixed["decoder_seed"])
        torch.nn.init.xavier_uniform_(model.w_rels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=fixed["finetune_lr"])
    for group in optimizer.param_groups:
        group["initial_lr"] = fixed["finetune_lr"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.8)
    stream = TrainingU(corpus, data.lookup, 103)
    historical = InnerSelection()
    history, best_weights, previous = [], None, 0
    if saved is not None and saved["stage"] == "finetune":
        previous, pretrain_updates = saved["cursor"], saved["pretrain_updates"]
        if not 1 <= previous <= epochs or pretrain_updates < 1:
            raise ValueError("Invalid TxGNN fine-tuning cursor")
        model.load_state_dict(saved["model"], strict=True)
        _restore_auxiliary(model, saved["auxiliary"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        stream.calls = saved["u_calls"]
        history, best_weights = saved["history"], saved["best_weights"]
        if not smoke:
            historical = InnerSelection.from_state(saved["inner_selection"])
        restore_rng(saved["rng"])
    for epoch in range(previous + 1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        contrast = native._contrast_graph(data, stream)
        loss, _ = adapter.loss(data.positive_graph, contrast)
        value = _update(native, model, optimizer, loss)
        scheduler.step(value)
        console.event("optimizer_update", stage="finetune", epoch=epoch, loss=value)
        if smoke or epoch in fixed["selection_epochs"]:
            metrics = _score(native, adapter, data, root, "support", panels["support"])
            key = selection_key(metrics, epoch)
            if (not history if smoke else historical.observe(epoch, metrics)):
                best_weights = model_state(model)
            history.append(dict(epoch=epoch, key=key))
            console.metrics(dict(support=metrics, contra={}))
            _write_checkpoint(root, binding, model, optimizer, stage="finetune", cursor=epoch, stream=stream,
                pretrain_updates=pretrain_updates, history=history, best_weights=best_weights,
                scheduler=scheduler.state_dict(), inner_selection=historical.state())
    selected = max(history, key=lambda row: tuple(row["key"]))["epoch"]
    model.load_state_dict(best_weights, strict=True)
    scores = {head: _score(native, adapter, data, root, head, members) for head, members in panels.items()}
    console.metrics(scores)
    save_training_state(root / "selected.pt", dict(binding=binding, epoch=selected, model=best_weights,
                                                  auxiliary=_capture_auxiliary(model)))
    write_json(root / "selection.json", dict(history=history, selected_epoch=selected))
    return scores, dict(updates=pretrain_updates + epochs, pretrain_updates=pretrain_updates, epochs=epochs,
                        selection="INNER_SUPPORT_AUC_AP_EARLIER_EPOCH", selected_epoch=selected)


def main():
    """Private JSON entry point for the isolated CPU interpreter."""
    import json
    from evi_hgt.settings import Settings
    from experiments.workflow.comparisons import run_comparison

    if sys.argv[1] == "--check":
        import dgl
        if dgl.__version__ != "1.1.2" or sys.version_info[:2] != (3, 11):
            raise RuntimeError("Use CPython 3.11 and DGL 1.1.2 from requirements-txgnn.txt")
        request = json.load(sys.stdin)
        native_txgnn(request["context"], smoke=request["smoke"])
        return

    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = run_comparison("TXGNN_ADAPTED", Settings(**request["settings"]), smoke=request["smoke"],
                            checkpoint=request["checkpoint"], context=request["context"])
    write_json(request["response"], dict(run=str(result)))


if __name__ == "__main__":
    main()
