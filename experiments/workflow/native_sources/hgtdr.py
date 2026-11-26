"""Portable HGTDR support and contra runs on the common signed development inputs."""

from contextlib import contextmanager
from functools import cache
from importlib import import_module
import random
import sys

import numpy as np
import torch

from evi_hgt.data.common import sha256
from evi_hgt.io import digest, write_json
from evi_hgt.runner import deterministic, restore_rng, rng_state
from experiments.workflow.comparisons import (
    load_training_state, model_state, save_training_state, scored_panel, selection_key,
)
from . import hgtdr_architecture
from .graph import native_graph, native_lookup, native_unlabeled
from .hgtdr_features import load_fixed_features

_ARCHIVE = "experiments.native_adapters.HGTDR.src"


@cache
def native_hgtdr():
    """Load the archived native CPU sampler with explicitly provided architecture imports."""
    archive_features = import_module(f"{_ARCHIVE}.rebuild_features")
    aliases = {"src.rebuild_features": archive_features, "src.runtime": hgtdr_architecture}
    previous_path = sys.path[:]
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        sys.modules.update(aliases)
        return import_module(f"{_ARCHIVE}.competition")
    except ImportError as error:
        raise RuntimeError("HGTDR requires a PyTorch-compatible torch-sparse build with the native hgt_sample operator") from error
    finally:
        sys.path[:] = previous_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@contextmanager
def _native_threads():
    previous = torch.get_num_threads()
    if previous != 4:
        torch.set_num_threads(4)
    try:
        yield
    finally:
        if previous != 4:
            torch.set_num_threads(previous)


class HGTDRFold:
    def __init__(self, corpus, context, root, *, smoke, device):
        self.native = native_hgtdr()
        self.seed, self.device = corpus.training_seed, device
        self.batch_size = 4096
        nodes, relations, arrays, graph_record = native_graph(corpus, context, root, smoke=smoke)
        self.features, self.available, feature_digest = load_fixed_features(corpus, nodes, context, smoke=smoke)
        self.lookup = native_lookup(nodes)
        counts = {name: len(values) for name, values in self.features.items()}
        self.sampler = self.native.QueryMaskedSampler(counts, relations, arrays)
        self.metadata = list(counts), self.sampler.edge_types
        self.unlabeled = native_unlabeled(corpus)
        self.identity = dict(corpus=corpus.identity, graph=digest(graph_record), extra_features=feature_digest,
                             torch=str(torch.__version__), sampler=sha256(self.native.__file__),
                             architecture=sha256(hgtdr_architecture.__file__))

    def pair_ids(self, pairs):
        drug, disease = self.lookup["drug"], self.lookup["disease"]
        return np.asarray([(drug[a], disease[b]) for a, b in pairs], dtype=np.int64)

    @_native_threads()
    def batch(self, pairs, seed):
        sample = self.sampler.sample(pairs, [512, 512, 512], seed=seed,
                                     max_nodes=20000, max_edges=1500000)
        sample["features"] = {kind: self.features[kind][ids].to(self.device)
                              for kind, ids in sample["nodes"].items()}
        sample["available"] = {kind: self.available[kind][ids].to(self.device)
                               for kind, ids in sample["nodes"].items()}
        sample["edges"] = {key: value.to(self.device) for key, value in sample["edges"].items()}
        sample["pair_index"] = sample["pair_index"].to(self.device)
        return sample

    def make_model(self, first_pairs):
        with deterministic():
            first = self.batch(first_pairs, self.native.seed_for(self.seed, "initialize"))
            return self.native.initialized_model(self, first, self.device, self.seed)

    def scores(self, model, members, head, *, domain=None):
        pairs = self.pair_ids((r["drug_id"], r["disease_id"]) for r in members)
        state, training = rng_state(), model.training
        scores = []
        domain = domain or ("validation" if head == "support" else "auxiliary_validation")
        try:
            model.eval()
            with torch.inference_mode(), deterministic():
                for offset in range(0, len(pairs), self.batch_size):
                    seed = self.native.seed_for(self.seed, "evaluation", domain, offset)
                    batch = self.batch(pairs[offset:offset + self.batch_size], seed)
                    values = model(batch).reshape(-1).cpu()
                    if not bool(torch.isfinite(values).all()):
                        raise FloatingPointError("Nonfinite HGTDR logit")
                    scores.extend(values.tolist())
        finally:
            restore_rng(state)
            model.train(training)
        if len(scores) != len(members):
            raise ValueError("HGTDR candidate score count differs")
        return scores


def _train_epoch(fold, model, optimizer, positives, epoch, epochs, console):
    seed_for = fold.native.seed_for
    positive_order = np.random.default_rng(seed_for(fold.seed, "positive-order", epoch)).permutation(len(positives))
    unlabeled = fold.pair_ids(fold.unlabeled.sample(len(positives), seed=seed_for(fold.seed, "unlabeled", epoch)))
    half = fold.batch_size // 2
    steps = 0
    with console.progress((len(positives) + half - 1) // half, f"Epoch {epoch}/{epochs}", unit="update") as progress:
        for batch_index, offset in enumerate(range(0, len(positive_order), half)):
            positive = positives[positive_order[offset:offset + half]]
            unknown = unlabeled[offset:offset + len(positive)]
            pair_rows = np.concatenate((positive, unknown))
            labels = torch.cat((torch.ones((len(positive), 1)), torch.zeros((len(unknown), 1)))).to(fold.device)
            with deterministic():
                batch = fold.batch(pair_rows, seed_for(fold.seed, "training-sample", epoch, batch_index))
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = fold.native.compute_loss(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite HGTDR training loss")
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not gradients or not all(bool(torch.isfinite(grad).all()) for grad in gradients):
                    raise FloatingPointError("Missing or nonfinite HGTDR gradients")
                optimizer.step()
            value = float(loss.detach())
            steps += 1
            console.event("optimizer_update", epoch=epoch, batch=batch_index, loss=value,
                          positive_pairs=len(positive), unlabeled_pairs=len(unknown))
            progress.update(loss=value)
    return steps


@_native_threads()
def run_hgtdr(method, head, corpus, panels, settings, root, console, *, smoke, checkpoint, context):
    """Fit the single-output native HGTDR predictor to one observed relation."""
    if not smoke and settings.device != "cuda":
        raise ValueError("The recorded HGTDR training recipe requires CUDA")
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the recorded native HGTDR recipe")
    fold = HGTDRFold(corpus, context, root, smoke=smoke, device=settings.device)
    positive_pairs = corpus.support if head == "support" else corpus.unsuitability
    positives = fold.pair_ids(sorted(positive_pairs))
    seed = corpus.training_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = fold.make_model(positives[:fold.batch_size])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0)
    epochs = 1 if smoke else 300
    binding = dict(method=method, head=head, data=fold.identity, device=settings.device,
                   epochs=epochs, batch_size=fold.batch_size,
                   panel_sha256=digest(panels[head]), recipe_sha256=sha256(__file__))
    updates, history, best_weights, previous = 0, [], None, 0
    if checkpoint:
        saved = load_training_state(checkpoint, binding)
        previous, updates, history, best_weights = (saved[key] for key in ("epoch", "updates", "history", "best_weights"))
        if not (1 <= previous <= epochs and len(history) == previous):
            raise ValueError("Invalid native HGTDR checkpoint epoch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
    for epoch in range(previous + 1, epochs + 1):
        updates += _train_epoch(fold, model, optimizer, positives, epoch, epochs, console)
        scheduler.step()
        metrics = scored_panel(root, head, panels[head], fold.scores(model, panels[head], head))
        key = selection_key(metrics, epoch)
        if not history or key > max(tuple(row["key"]) for row in history):
            best_weights = model_state(model)
        history.append(dict(epoch=epoch, key=key))
        console.metrics({head: metrics, ("contra" if head == "support" else "support"): {}})
        save_training_state(root / "checkpoint.pt", dict(binding=binding, epoch=epoch, updates=updates, history=history,
            model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            rng=rng_state(), best_weights=best_weights))
    selected_epoch = max(history, key=lambda row: tuple(row["key"]))["epoch"]
    model.load_state_dict(best_weights, strict=True)
    metrics = scored_panel(root, head, panels[head], fold.scores(model, panels[head], head))
    save_training_state(root / "selected.pt", dict(binding=binding, model=best_weights, epoch=selected_epoch))
    write_json(root / "selection.json", dict(history=history, selected_epoch=selected_epoch))
    return {head: metrics}, dict(updates=updates, epochs=epochs, selection="INNER_AUC_AP_EARLIER_EPOCH",
                                 selected_epoch=selected_epoch)
