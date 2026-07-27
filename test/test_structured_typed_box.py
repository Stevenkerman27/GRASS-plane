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
import cst_airfoil_codec
from grassdata import StructuredGRASSDataset, Tree
from torchfoldext import FoldExt


def build_structured_sample():
    code = cst_airfoil_codec.pack_cst_airfoil_code(
        torch.full((util.CST_SURFACE_SHAPE_COEFFICIENTS,), 0.12),
        torch.full((util.CST_SURFACE_SHAPE_COEFFICIENTS,), -0.10),
        upper_trailing_edge_y=0.0,
        lower_trailing_edge_y=0.0,
        class_function_n1=0.5,
        class_function_n2=1.0,
    )
    sections = torch.zeros(util.MAX_SECTION_COUNT, util.COMPONENT_SECTION_SIZES[util.COMPONENT_WING])
    sections[:2, :util.CST_AIRFOIL_CODE_SIZE] = code
    sections[0, 24:29] = torch.tensor([0.25, 0.0, 0.0, 0.30, 0.0])
    sections[1, 24:29] = torch.tensor([0.25, 0.5, 0.0, 0.15, 0.0])
    return {
        "boxes": [
            {
                "component": util.COMPONENT_FUSELAGE,
                "sections": torch.tensor([
                    [0.0, 0.0, 0.0, 0.08, 0.10],
                    [0.4, 0.0, 0.0, 0.14, 0.15],
                    [0.8, 0.0, 0.0, 0.06, 0.08],
                    *([[0.0] * util.FUSELAGE_SECTION_SIZE] * (util.MAX_SECTION_COUNT - 3)),
                ]),
                "section_count": 3,
            },
            {
                "component": util.COMPONENT_WING,
                "sections": sections,
                "section_count": 2,
            },
            {
                "component": util.COMPONENT_ENGINE,
                "geometry": torch.full((util.ENGINE_GEOMETRY_SIZE,), 0.25),
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

    wing_grad = decoder.wing_section_decoder.output.weight.grad
    fuselage_grad = decoder.fuselage_section_decoder.output.weight.grad
    component_grad = decoder.component_classifier.mlp2.weight.grad
    assert wing_grad is not None
    assert fuselage_grad is not None
    assert component_grad is not None


def test_sequence_models_use_classic_rnn_and_fuselage_constraints():
    config = build_config()
    encoder = grassmodel.GRASSEncoder(config)
    decoder = grassmodel.GRASSDecoder(config)
    teacher_sections = torch.zeros(
        1, util.MAX_SECTION_COUNT, util.FUSELAGE_SECTION_SIZE
    )
    sections, count_logits = decoder.fuselageSectionDecoder(
        torch.zeros(1, config.feature_size), teacher_sections
    )

    assert isinstance(encoder.fuselage_section_encoder.rnn, torch.nn.RNN)
    assert isinstance(encoder.wing_section_encoder.rnn, torch.nn.RNN)
    assert isinstance(decoder.fuselage_section_decoder.rnn, torch.nn.RNNCell)
    assert sections.shape == (1, util.MAX_SECTION_COUNT, util.FUSELAGE_SECTION_SIZE)
    assert count_logits.shape == (
        1, util.MAX_SECTION_COUNT - util.MIN_SECTION_COUNT + 1
    )
    assert torch.equal(sections[:, 0, 0], torch.zeros(1))
    assert torch.all(sections[:, 1:, 0] > sections[:, :-1, 0])
    assert torch.all(sections[:, :, 3:5] > 0.0)


def test_structured_wing_requires_sections():
    sample = build_structured_sample()
    del sample["boxes"][1]["sections"]

    with pytest.raises(KeyError, match="missing required key"):
        Tree.from_structured_sample(sample)


def test_structured_wing_requires_padded_29d_sections():
    sample = build_structured_sample()
    sample["boxes"][1]["sections"] = torch.zeros(util.MAX_SECTION_COUNT - 1, util.COMPONENT_SECTION_SIZES[util.COMPONENT_WING])

    with pytest.raises(ValueError, match="must have shape"):
        Tree.from_structured_sample(sample)


def test_structured_engine_rejects_sections():
    sample = build_structured_sample()
    sample["boxes"][2]["sections"] = torch.zeros(util.MAX_SECTION_COUNT, util.FUSELAGE_SECTION_SIZE)

    with pytest.raises(KeyError, match="must not define sequence sections"):
        Tree.from_structured_sample(sample)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
