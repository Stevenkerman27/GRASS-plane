import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import util


SEQUENCE_TYPE_FUSELAGE = 'fuselage'
SEQUENCE_TYPE_WING = 'wing'
SEQUENCE_SPECS = {
    SEQUENCE_TYPE_FUSELAGE: {
        'component': util.COMPONENT_FUSELAGE,
        'global_size': util.FUSELAGE_GLOBAL_SIZE,
        'section_size': util.FUSELAGE_SECTION_SIZE,
        'section_count': util.FUSELAGE_SECTION_COUNT,
    },
    SEQUENCE_TYPE_WING: {
        'component': util.COMPONENT_WING,
        'global_size': util.WING_GLOBAL_SIZE,
        'section_size': util.WING_SECTION_SIZE,
        'section_count': util.WING_SECTION_COUNT,
    },
}
AE_COMPONENT_TYPES = (util.COMPONENT_FUSELAGE, util.COMPONENT_WING)
AE_COMPONENT_CLASS_SIZE = len(AE_COMPONENT_TYPES)


def sequence_spec(sequence_type):
    if sequence_type not in SEQUENCE_SPECS:
        raise ValueError(f'Unknown sequence type: {sequence_type}')
    return SEQUENCE_SPECS[sequence_type]


def sequence_max_sections(sequence_type):
    return sequence_spec(sequence_type)['section_count']


def sequence_section_size(sequence_type):
    return sequence_spec(sequence_type)['section_size']


def sequence_global_size(sequence_type):
    return sequence_spec(sequence_type)['global_size']


def sequence_type_for_component(component):
    for sequence_type, spec in SEQUENCE_SPECS.items():
        if spec['component'] == component:
            return sequence_type
    raise ValueError(f'Component {component} is not a structured sequence component')


class Tree(object):
    class NodeType(Enum):
        BOX = 0  # box node
        ADJ = 1  # adjacency (adjacent part assembly) node
        SYM = 2  # symmetry (symmetric part grouping) node

    class Node(object):
        def __init__(self, box=None, left=None, right=None, node_type=None, sym=None):
            self.box = box          # box feature vector for a leaf node
            self.sym = sym          # symmetry parameter vector for a symmetry node
            self.left = left        # left child for ADJ or SYM (a symmeter generator)
            self.right = right      # right child
            self.node_type = node_type
            self.label = torch.LongTensor([self.node_type.value])

        def is_leaf(self):
            return self.node_type == Tree.NodeType.BOX and self.box is not None

        def is_adj(self):
            return self.node_type == Tree.NodeType.ADJ

        def is_sym(self):
            return self.node_type == Tree.NodeType.SYM

    def __init__(self, boxes, ops, syms): #boxes: torch vector [M,12], syms torch vector [N,8]
        box_list = [b for b in torch.split(boxes, 1, 0)] #切成M个[1, 12]
        sym_param = [s for s in torch.split(syms, 1, 0)]
        self.root = self._build_root_from_postorder(box_list, ops, sym_param)

    @classmethod
    def from_structured_sample(cls, sample):
        boxes, ops, syms = validate_structured_sample(sample)
        tree = cls.__new__(cls)
        tree.root = cls._build_root_from_postorder(boxes, ops, syms)
        return tree

    @staticmethod
    def _build_root_from_postorder(box_list, ops, sym_param):
        box_list.reverse()
        sym_param.reverse()
        queue = []
        for op_value in torch.reshape(ops, (-1,)).tolist():
            if op_value == Tree.NodeType.BOX.value:
                queue.append(Tree.Node(box=box_list.pop(), node_type=Tree.NodeType.BOX)) #读到box，把box添加入栈
            elif op_value == Tree.NodeType.ADJ.value:
                left_node = queue.pop()
                right_node = queue.pop() #跳出两个node,放入一个ADJ node
                queue.append(Tree.Node(left=left_node, right=right_node, node_type=Tree.NodeType.ADJ))
            elif op_value == Tree.NodeType.SYM.value:
                node = queue.pop()
                queue.append(Tree.Node(left=node, sym=sym_param.pop(), node_type=Tree.NodeType.SYM))
            elif op_value == -1:
                continue
            else:
                raise ValueError(f"Unknown tree op: {op_value}")
        assert len(queue) == 1
        return queue[0]


def _as_row_tensor(value, expected_size, field_name):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.size(0) != 1 or tensor.size(1) != expected_size:
        raise ValueError(
            f"{field_name} must have shape [1, {expected_size}], "
            f"got {list(tensor.size())}"
        )
    return tensor


def _sequence_type_from_box(box, component, box_index):
    if "sequence_type" not in box:
        raise KeyError(f'boxes[{box_index}] missing required key: sequence_type')
    sequence_type = box["sequence_type"]
    spec = sequence_spec(sequence_type)
    if spec['component'] != component:
        raise ValueError(
            f'boxes[{box_index}].sequence_type {sequence_type!r} does not match '
            f'component {util.component_name(component)!r}'
        )
    return sequence_type


def _as_component_sections(value, sequence_type, field_name):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    expected_shape = (
        sequence_max_sections(sequence_type),
        sequence_section_size(sequence_type),
    )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f'{field_name} must have shape {list(expected_shape)}, got {list(tensor.shape)}')
    if not torch.isfinite(tensor).all():
        raise ValueError(f'{field_name} must contain only finite values')
    return tensor.unsqueeze(0)


def _as_component_global(value, sequence_type, field_name):
    return _as_row_tensor(value, sequence_global_size(sequence_type), field_name)


def _component_from_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"component tensor must contain exactly one value, got {value.numel()}")
        value = int(value.item())
    else:
        value = int(value)
    util.component_name(value)
    return value


def _validate_structured_box(box, box_index):
    if not isinstance(box, dict):
        raise TypeError(f"boxes[{box_index}] must be dict, got {type(box).__name__}")
    if "component" not in box:
        raise KeyError(f"boxes[{box_index}] missing required key: {"component"}")

    component = _component_from_value(box["component"])
    if component in AE_COMPONENT_TYPES:
        sequence_type = _sequence_type_from_box(box, component, box_index)
        for key in ("z_global", "z_section"):
            if key not in box:
                name = util.component_name(component)
                raise KeyError(f"{name} boxes[{box_index}] missing required key: {key}")
        if "geometry" in box or "sections" in box:
            name = util.component_name(component)
            raise KeyError(f"{name} boxes[{box_index}] must not define {"geometry"}")
        if "section_count" in box:
            section_count = torch.as_tensor(box["section_count"], dtype=torch.long).reshape(-1)
            expected_count = sequence_max_sections(sequence_type)
            if section_count.numel() != 1 or int(section_count.item()) != expected_count:
                name = util.component_name(component)
                raise ValueError(
                    f"{name} boxes[{box_index}].section_count must equal "
                    f"{expected_count}"
                )
        structured_box = {
            "component": component,
            "sequence_type": sequence_type,
            "z_global": _as_component_global(
                box["z_global"], sequence_type, f"boxes[{box_index}].z_global"
            ),
            "sections": _as_component_sections(
                box["z_section"], sequence_type, f"boxes[{box_index}].z_section"
            ),
        }
    else:
        if "geometry" not in box:
            raise KeyError(f"boxes[{box_index}] missing required key: {"geometry"}")
        if "sections" in box or "section_count" in box:
            name = util.component_name(component)
            raise KeyError(f"{name} boxes[{box_index}] must not define sequence sections")
        geometry_size = util.COMPONENT_GEOMETRY_SIZES[component]
        structured_box = {
            "component": component,
            "geometry": _as_row_tensor(
                box["geometry"],
                geometry_size,
                f"boxes[{box_index}].{"geometry"}",
            ),
        }
    if "sections" in box and component not in AE_COMPONENT_TYPES:
        name = util.component_name(component)
        raise KeyError(f"{name} boxes[{box_index}] must not define {"sections"}")

    return structured_box


def validate_structured_sample(sample):
    if not isinstance(sample, dict):
        raise TypeError(f"structured sample must be dict, got {type(sample).__name__}")
    for key in ('boxes', 'ops', 'syms'):
        if key not in sample:
            raise KeyError(f"structured sample missing required key: {key}")

    boxes = [
        _validate_structured_box(box, index)
        for index, box in enumerate(sample['boxes'])
    ]
    ops = torch.as_tensor(sample['ops'], dtype=torch.int64)
    if ops.dim() == 0:
        raise ValueError("ops must contain at least one operation")
    op_values = torch.reshape(ops, (-1,))
    box_count = int(torch.sum(op_values == Tree.NodeType.BOX.value).item())
    sym_count = int(torch.sum(op_values == Tree.NodeType.SYM.value).item())
    if box_count != len(boxes):
        raise ValueError(f"ops contain {box_count} BOX entries, but sample has {len(boxes)} boxes")

    syms_tensor = torch.as_tensor(sample['syms'], dtype=torch.float32)
    if syms_tensor.numel() == 0:
        syms_tensor = syms_tensor.reshape(0, 8)
    elif syms_tensor.dim() == 1:
        syms_tensor = syms_tensor.unsqueeze(0)
    if syms_tensor.dim() != 2 or syms_tensor.size(1) != 8:
        raise ValueError(f"syms must have shape [N, 8], got {list(syms_tensor.size())}")
    if sym_count != syms_tensor.size(0):
        raise ValueError(f"ops contain {sym_count} SYM entries, but sample has {syms_tensor.size(0)} syms")
    syms = [s for s in torch.split(syms_tensor, 1, 0)]

    return boxes, ops, syms


class GRASSDataset(data.Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        box_data = torch.from_numpy(loadmat(self.dir+'/box_data.mat')['boxes']).float()
        op_data = torch.from_numpy(loadmat(self.dir+'/op_data.mat')['ops']).int()
        sym_data = torch.from_numpy(loadmat(self.dir+'/sym_data.mat')['syms']).float()
        #weight_list = torch.from_numpy(loadmat(self.dir+'/weights.mat')['weights']).float()
        num_examples = op_data.size()[1] #样本数
        box_data = torch.chunk(box_data, num_examples, 1) #切成num_example个独立tensor
        op_data = torch.chunk(op_data, num_examples, 1) #读取正确的合并顺序
        sym_data = torch.chunk(sym_data, num_examples, 1)
        #weight_list = torch.chunk(weight_list, num_examples, 1)
        self.transform = transform
        self.trees = []
        for i in range(len(op_data)) :
            boxes = torch.t(box_data[i])
            ops = torch.t(op_data[i])
            syms = torch.t(sym_data[i])
            tree = Tree(boxes, ops, syms)
            self.trees.append(tree)

    def __getitem__(self, index):
        tree = self.trees[index]
        return tree

    def __len__(self):
        return len(self.trees)


class StructuredGRASSDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        samples = torch.load(self.path, map_location='cpu', weights_only=True)
        if not isinstance(samples, list):
            raise TypeError(f"structured dataset must be a list, got {type(samples).__name__}")
        if len(samples) == 0:
            raise ValueError(f"structured dataset is empty: {self.path}")
        self.transform = transform
        self.trees = [
            Tree.from_structured_sample(sample)
            for sample in samples
        ]

    def __getitem__(self, index):
        tree = self.trees[index]
        if self.transform is not None:
            tree = self.transform(tree)
        return tree

    def __len__(self):
        return len(self.trees)
