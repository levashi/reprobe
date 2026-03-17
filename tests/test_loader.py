import os
import tempfile
import torch
from reprobe import ProbeLoader, ProbesTrainer


def compare_probes(orig, new):
    assert orig.meta == new.meta
    assert torch.allclose(orig.model[0].weight.data, new.model[0].weight.data)


def test_probe_single_mode():
    hidden_dim = 16
    size = 100
    acts = torch.full((size, 3, hidden_dim), 4).float()
    labels = torch.cat([torch.ones(int(size/2)), torch.zeros(int(size/2))])
    acts_dict = {"prefill": acts, "token": None}
    labels_dict = {"prefill": labels, "token": None}

    mode = "prefill"
    model_id = "test"
    trainer = ProbesTrainer(model_id, hidden_dim)
    trainer.train_probes(acts_dict, labels_dict, ["test"], epochs=1, training_mode=mode)
    probes = trainer.probes

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save(tmpdir, one_file=True)
        loaded = ProbeLoader.from_file(os.path.join(tmpdir, f"{model_id}_probes.pt"))
        for layer in probes[mode]:
            compare_probes(probes[mode][layer], loaded[mode][layer])

        trainer.save(tmpdir)
        loaded = ProbeLoader.from_registry(os.path.join(tmpdir, "registry.json"))
        for layer in probes[mode]:
            compare_probes(probes[mode][layer], loaded[mode][layer])


def test_probe_all_mode():
    hidden_dim = 16
    size = 100
    acts = torch.full((size, 3, hidden_dim), 4).float()
    labels = torch.cat([torch.ones(int(size/2)), torch.zeros(int(size/2))])
    acts_dict = {"prefill": acts, "token": acts}
    labels_dict = {"prefill": labels, "token": labels}

    model_id = "test"
    trainer = ProbesTrainer(model_id, hidden_dim)
    trainer.train_probes(acts_dict, labels_dict, ["test"], epochs=1, training_mode="all")
    probes = trainer.probes

    # Vérifier que les deux modes ont bien été entraînés
    assert len(probes["prefill"]) > 0, "Aucune probe prefill entraînée"
    assert len(probes["token"]) > 0, "Aucune probe token entraînée"

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save(tmpdir, one_file=True)
        loaded = ProbeLoader.from_file(os.path.join(tmpdir, f"{model_id}_probes.pt"))
        
        for mode in ["prefill", "token"]:
            for layer in probes[mode]:
                compare_probes(probes[mode][layer], loaded[mode][layer])

        trainer.save(tmpdir)
        loaded = ProbeLoader.from_registry(os.path.join(tmpdir, "registry.json"))

        for mode in ["prefill", "token"]:
            for layer in probes[mode]:
                compare_probes(probes[mode][layer], loaded[mode][layer])