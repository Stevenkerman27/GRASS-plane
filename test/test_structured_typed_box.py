from argparse import Namespace
from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import grassmodel
import util
from grassdata import StructuredGRASSDataset, Tree
from torchfoldext import FoldExt


def build_structured_sample():
    return {
        "boxes": [
            {
                util.BOX_COMPONENT_KEY: util.COMPONENT_FUSELAGE,
                util.BOX_GEOMETRY_KEY: torch.zeros(util.FUSELAGE_GEOMETRY_SIZE),
            },
            {
                util.BOX_COMPONENT_KEY: util.COMPONENT_WING,
                util.BOX_GEOMETRY_KEY: torch.ones(util.WING_GEOMETRY_SIZE),
                util.BOX_AIRFOIL_KEY: torch.zeros(util.AIRFOIL_BEZIER_CODE_SIZE),
            },
            {
                util.BOX_COMPONENT_KEY: util.COMPONENT_ENGINE,
                util.BOX_GEOMETRY_KEY: torch.full((util.ENGINE_GEOMETRY_SIZE,), 0.25),
            },
        ],
        "ops": torch.tensor([
            Tree.NodeType.BOX.value,
            Tree.NodeType.BOX.value,
            Tree.NodeType.ADJ.value,
            Tree.NodeType.BOX.value,
            Tree.NodeType.ADJ.value,
        ]),
        "syms": torch.empty(0, 8),
    }


def build_config():
    return Namespace(
        box_code_size=13,
        feature_size=16,
        hidden_size=32,
        symmetry_size=8,
    )


def test_structured_dataset_and_typed_encoder_forward(tmp_path):
    dataset_path = tmp_path / "structured.pt"
    torch.save([build_structured_sample()], dataset_path)

    dataset = StructuredGRASSDataset(dataset_path)
    tree = dataset[0]

    encoder = grassmodel.GRASSEncoder(build_config())
    fold = FoldExt(cuda=False)
    encoded_node = grassmodel.encode_structure_fold(fold, tree, use_sampler=False)
    encoded = fold.apply(encoder, [[encoded_node]])[0]

    assert encoded.shape == (1, 16)


def test_structured_typed_decoder_loss_backward(tmp_path):
    dataset_path = tmp_path / "structured.pt"
    torch.save([build_structured_sample()], dataset_path)

    tree = StructuredGRASSDataset(dataset_path)[0]
    config = build_config()
    encoder = grassmodel.GRASSEncoder(config)
    decoder = grassmodel.GRASSDecoder(config)

    enc_fold = FoldExt(cuda=False)
    encoded_node = grassmodel.encode_structure_fold(enc_fold, tree)
    encoded = enc_fold.apply(encoder, [[encoded_node]])[0]
    root_code, _ = torch.chunk(encoded, 2, 1)

    dec_fold = FoldExt(cuda=False)
    box_nodes, sym_nodes, cat_nodes = grassmodel.decode_structure_fold(dec_fold, root_code, tree)
    assert len(box_nodes) == 3
    assert len(sym_nodes) == 0
    assert len(cat_nodes) == 5

    box_loss, cat_loss = dec_fold.apply(decoder, [box_nodes, cat_nodes])
    total_loss = box_loss.sum() + cat_loss.sum()
    total_loss.backward()

    wing_grad = decoder.wing_box_decoder.mlp.weight.grad
    component_grad = decoder.component_classifier.mlp2.weight.grad
    assert wing_grad is not None
    assert component_grad is not None


def test_structured_wing_requires_airfoil():
    sample = build_structured_sample()
    del sample["boxes"][1][util.BOX_AIRFOIL_KEY]

    with pytest.raises(KeyError, match="missing required key"):
        Tree.from_structured_sample(sample)


def test_structured_non_wing_rejects_airfoil():
    sample = build_structured_sample()
    sample["boxes"][0][util.BOX_AIRFOIL_KEY] = torch.zeros(util.AIRFOIL_BEZIER_CODE_SIZE)

    with pytest.raises(KeyError, match="must not define"):
        Tree.from_structured_sample(sample)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
