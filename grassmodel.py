import math
import torch
from torch import nn
from torch.nn import functional as F
from time import time
import util
import grassdata
import section_parameter_codec
from grassdata import Tree

SEQUENCE_RNN_LAYERS = 1

#########################################################################################
## Encoder
#########################################################################################

class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        encode = self.tanh(self.mlp1(input))
        mu = self.mlp2mu(encode)
        logvar = self.mlp2var(encode)
        std = logvar.mul(0.5).exp_() # calculate the STDEV
        eps = torch.randn_like(std) # random normalized noise
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        return torch.cat([eps.mul(std).add_(mu), KLD_element], 1)

class BoxEncoder(nn.Module):

    def __init__(self, input_size, feature_size):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, box_input):
        box_vector = self.encoder(box_input)
        box_vector = self.tanh(box_vector)
        return box_vector


class SectionEncoder(nn.Module):
    def __init__(self, sequence_type, feature_size, rnn_type, parameter_statistics):
        super(SectionEncoder, self).__init__()
        self.sequence_type = sequence_type
        self.section_size = grassdata.sequence_section_size(sequence_type)
        self.max_sections = grassdata.sequence_max_sections(sequence_type)
        self.rnn_type = util.validate_ae_rnn_type(rnn_type)
        self.parameter_codec = section_parameter_codec.SectionParameterCodec(
            sequence_type, parameter_statistics
        )
        recurrent_class = nn.RNN if self.rnn_type == 'rnn' else nn.GRU
        recurrent_kwargs = {
            'input_size': self.section_size,
            'hidden_size': feature_size,
            'batch_first': True,
            'num_layers': SEQUENCE_RNN_LAYERS,
        }
        if self.rnn_type == 'rnn':
            recurrent_kwargs['nonlinearity'] = 'tanh'
        self.rnn = recurrent_class(**recurrent_kwargs)

    def forward(self, sections, section_count):
        expected_shape = (self.max_sections, self.section_size)
        if tuple(sections.shape[1:]) != expected_shape:
            raise ValueError(f'sections must have shape [B, {expected_shape[0]}, {expected_shape[1]}]')
        counts = section_count.reshape(-1).to(dtype=torch.long, device=sections.device)
        grassdata.section_mask(counts, sequence_type=self.sequence_type)
        model_sections = self.parameter_codec.normalize(sections, counts)
        packed = nn.utils.rnn.pack_padded_sequence(
            model_sections, counts.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        return hidden.squeeze(0)

class AdjEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(AdjEncoder, self).__init__()
        self.left = nn.Linear(feature_size, hidden_size)
        self.right = nn.Linear(feature_size, hidden_size, bias=False)
        self.second = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, left_input, right_input):
        output = self.left(left_input)
        output += self.right(right_input)
        output = self.tanh(output)
        output = self.second(output)
        output = self.tanh(output)
        return output

class SymEncoder(nn.Module):

    def __init__(self, feature_size, symmetry_size, hidden_size):
        super(SymEncoder, self).__init__()
        self.left = nn.Linear(feature_size, hidden_size)
        self.right = nn.Linear(symmetry_size, hidden_size)
        self.second = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, left_input, right_input):
        output = self.left(left_input)
        output += self.right(right_input)
        output = self.tanh(output)
        output = self.second(output)
        output = self.tanh(output)
        return output

class GRASSEncoder(nn.Module):

    def __init__(self, config, section_statistics):
        super(GRASSEncoder, self).__init__()
        self.box_encoder = BoxEncoder(input_size = config.box_code_size, feature_size = config.feature_size)
        self.engine_obb_encoder = BoxEncoder(input_size=util.OBB_GEOMETRY_SIZE, feature_size=config.feature_size)
        self.fuselage_section_encoder = SectionEncoder(
            grassdata.SEQUENCE_TYPE_FUSELAGE,
            config.feature_size,
            config.ae_rnn_type,
            section_statistics[grassdata.SEQUENCE_TYPE_FUSELAGE],
        )
        self.wing_section_encoder = SectionEncoder(
            grassdata.SEQUENCE_TYPE_WING,
            config.feature_size,
            config.ae_rnn_type,
            section_statistics[grassdata.SEQUENCE_TYPE_WING],
        )
        self.adj_encoder = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.sym_encoder = SymEncoder(feature_size = config.feature_size, symmetry_size = config.symmetry_size, hidden_size = config.hidden_size)
        self.sample_encoder = Sampler(feature_size = config.feature_size, hidden_size = config.hidden_size)

    def boxEncoder(self, box):
        return self.box_encoder(box)

    def obbBoxEncoder(self, geometry):
        return self.engine_obb_encoder(geometry)

    def fuselageSectionEncoder(self, sections, section_count):
        return self.fuselage_section_encoder(sections, section_count)

    def wingSectionEncoder(self, sections, section_count):
        return self.wing_section_encoder(sections, section_count)

    def adjEncoder(self, left, right):
        return self.adj_encoder(left, right)

    def symEncoder(self, feature, sym):
        return self.sym_encoder(feature, sym)

    def sampleEncoder(self, feature):
        return self.sample_encoder(feature)

def encode_structure_fold(fold, tree, use_sampler=True):
    """
    Encodes a structure into a feature vector.
    If use_sampler is True (default), it includes the final Sampler step (mu/logvar).
    If use_sampler is False, it returns the raw root feature, used by the Discriminator.
    """
    def encode_node(node):
        if node.is_leaf():
            if isinstance(node.box, dict):
                sequence_type = node.box["sequence_type"]
                if sequence_type == grassdata.SEQUENCE_TYPE_WING:
                    return fold.add(
                        'wingSectionEncoder',
                        node.box["sections"],
                        node.box["section_count"],
                    )
                if sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
                    return fold.add(
                        'fuselageSectionEncoder',
                        node.box["sections"],
                        node.box["section_count"],
                    )
                if node.box['component'] == util.COMPONENT_ENGINE:
                    return fold.add('obbBoxEncoder', node.box["geometry"])
                raise ValueError(f"Unknown sequence type: {sequence_type}")
            return fold.add('boxEncoder', node.box) #fold.add只录制，不执行
        elif node.is_adj():
            left = encode_node(node.left)
            right = encode_node(node.right)
            return fold.add('adjEncoder', left, right)
        elif node.is_sym():
            feature = encode_node(node.left)
            sym = node.sym
            return fold.add('symEncoder', feature, sym)

    encoding = encode_node(tree.root) #递归的合并根节点
    if use_sampler:
        return fold.add('sampleEncoder', encoding)
    else:
        return encoding

#########################################################################################
## Decoder
#########################################################################################

class NodeClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(hidden_size, 3)
        #self.softmax = nn.Softmax()

    def forward(self, input_feature):
        output = self.mlp1(input_feature)
        output = self.tanh(output)
        output = self.mlp2(output)
        #output = self.softmax(output)
        return output


class ComponentClassifier(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(ComponentClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, grassdata.AE_COMPONENT_CLASS_SIZE)

    def forward(self, input_feature):
        return self.mlp2(torch.tanh(self.mlp1(input_feature)))

class SampleDecoder(nn.Module):
    """ Decode a randomly sampled noise into a feature vector """
    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()
        
    def forward(self, input_feature):
        output = self.tanh(self.mlp1(input_feature))
        output = self.tanh(self.mlp2(output))
        return output

class AdjDecoder(nn.Module):
    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self, feature_size, hidden_size):
        super(AdjDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.mlp_left = nn.Linear(hidden_size, feature_size)
        self.mlp_right = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        left_feature = self.mlp_left(vector)
        left_feature = self.tanh(left_feature)
        right_feature = self.mlp_right(vector)
        right_feature = self.tanh(right_feature)
        return left_feature, right_feature

class SymDecoder(nn.Module):

    def __init__(self, feature_size, symmetry_size, hidden_size):
        super(SymDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size) # layer for decoding a feature vector 
        self.tanh = nn.Tanh()
        self.mlp_sg = nn.Linear(hidden_size, feature_size) # layer for outputing the feature of symmetry generator
        self.mlp_sp = nn.Linear(hidden_size, symmetry_size) # layer for outputing the vector of symmetry parameter

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        sym_gen_vector = self.mlp_sg(vector)
        sym_gen_vector = self.tanh(sym_gen_vector)
        sym_param_vector = self.mlp_sp(vector)
        sym_param_vector = self.tanh(sym_param_vector)
        return sym_gen_vector, sym_param_vector

class BoxDecoder(nn.Module):

    def __init__(self, feature_size, box_size):
        super(BoxDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, box_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        import torch
        vector = self.mlp(parent_feature)
        vector_geom = self.tanh(vector[:, :10])
        vector_cls = vector[:, 10:]
        return torch.cat([vector_geom, vector_cls], dim=1)


class TypedBoxDecoder(nn.Module):
    def __init__(self, feature_size, output_size, tanh_size):
        super(TypedBoxDecoder, self).__init__()
        if tanh_size > output_size:
            raise ValueError(f"tanh_size {tanh_size} exceeds output_size {output_size}")
        self.mlp = nn.Linear(feature_size, output_size)
        self.tanh_size = tanh_size
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector_tanh = self.tanh(vector[:, :self.tanh_size])
        vector_raw = vector[:, self.tanh_size:]
        return torch.cat([vector_tanh, vector_raw], dim=1)


class AutoregressiveSectionDecoder(nn.Module):
    def __init__(self, sequence_type, feature_size, rnn_type, parameter_statistics):
        super(AutoregressiveSectionDecoder, self).__init__()
        self.sequence_type = sequence_type
        self.component = grassdata.sequence_spec(sequence_type)['component']
        self.section_size = grassdata.sequence_section_size(sequence_type)
        self.max_sections = grassdata.sequence_max_sections(sequence_type)
        self.minimum_sections = grassdata.sequence_section_count_range(sequence_type)[0]
        self.rnn_type = util.validate_ae_rnn_type(rnn_type)
        self.feature_size = feature_size
        self.parameter_codec = section_parameter_codec.SectionParameterCodec(
            sequence_type, parameter_statistics
        )
        self.bos = nn.Parameter(torch.zeros(self.section_size))
        self.initial_hidden = nn.Linear(feature_size, feature_size)
        recurrent_cell_class = (
            nn.RNNCell if self.rnn_type == 'rnn' else nn.GRUCell
        )
        recurrent_cell_kwargs = {}
        if self.rnn_type == 'rnn':
            recurrent_cell_kwargs['nonlinearity'] = 'tanh'
        self.rnn = recurrent_cell_class(
            self.section_size + feature_size + 1,
            feature_size,
            **recurrent_cell_kwargs,
        )
        self.output = nn.Linear(feature_size, self.section_size)
        self.count = nn.Linear(
            feature_size, self.max_sections - self.minimum_sections + 1
        )

    def _canonicalize_model_step(self, model_section, index):
        if self.component == util.COMPONENT_FUSELAGE and index == 0:
            model_section = model_section.clone()
            model_section[:, 0] = 0.0
        return model_section

    def _step_input(self, previous, parent_feature, index):
        normalized_step = parent_feature.new_full(
            (parent_feature.size(0), 1), index / (self.max_sections - 1)
        )
        return torch.cat([previous, parent_feature, normalized_step], dim=1)

    def forward(
            self, parent_feature, teacher_sections, teacher_count,
            teacher_forcing_probability):
        expected_shape = (self.max_sections, self.section_size)
        if tuple(teacher_sections.shape[1:]) != expected_shape:
            raise ValueError(
                f'teacher sections must have shape [B, {expected_shape[0]}, {expected_shape[1]}]'
            )
        probability = torch.as_tensor(
            teacher_forcing_probability, dtype=parent_feature.dtype, device=parent_feature.device
        ).reshape(-1)
        if probability.numel() == 1:
            probability = probability.expand(parent_feature.size(0))
        if probability.numel() != parent_feature.size(0):
            raise ValueError('teacher_forcing_probability must have one value per batch sample.')
        if torch.any(probability < 0.0) or torch.any(probability > 1.0):
            raise ValueError('teacher_forcing_probability must be in [0, 1].')
        teacher_model_sections = self.parameter_codec.normalize(
            teacher_sections, teacher_count
        )
        hidden = torch.tanh(self.initial_hidden(parent_feature))
        previous = self.bos.unsqueeze(0).expand(parent_feature.size(0), -1)
        model_sections = []
        for index in range(self.max_sections):
            hidden = self.rnn(self._step_input(previous, parent_feature, index), hidden)
            model_section = self._canonicalize_model_step(self.output(hidden), index)
            model_sections.append(model_section)
            if torch.all(probability == 1.0):
                previous = teacher_model_sections[:, index, :]
            elif torch.all(probability == 0.0):
                previous = model_section
            else:
                choose_teacher = torch.rand(
                    (parent_feature.size(0), 1), device=parent_feature.device
                ) < probability.unsqueeze(1)
                previous = torch.where(
                    choose_teacher, teacher_model_sections[:, index, :], model_section
                )
        physical_sections = self.parameter_codec.denormalize(
            torch.stack(model_sections, dim=1)
        )
        return physical_sections, self.count(parent_feature)

    def generate(self, parent_feature):
        """Generate padded sections without ground-truth feedback.

        The decoder always emits MAX_SECTION_COUNT sections for batching.  The
        predicted count is the sole authority for the returned valid mask.
        """
        if parent_feature.dim() != 2:
            raise ValueError('parent_feature must have shape [B, feature_size]')
        hidden = torch.tanh(self.initial_hidden(parent_feature))
        previous = self.bos.unsqueeze(0).expand(parent_feature.size(0), -1)
        model_sections = []
        for index in range(self.max_sections):
            hidden = self.rnn(self._step_input(previous, parent_feature, index), hidden)
            model_section = self._canonicalize_model_step(self.output(hidden), index)
            model_sections.append(model_section)
            previous = model_section
        generated_sections = self.parameter_codec.denormalize(
            torch.stack(model_sections, dim=1)
        )
        count_logits = self.count(parent_feature)
        section_count = torch.argmax(count_logits, dim=1) + self.minimum_sections
        valid_mask = grassdata.section_mask(
            section_count, device=parent_feature.device, sequence_type=self.sequence_type
        )
        return generated_sections, section_count, valid_mask, count_logits

def wing_section_reconstruction_losses(
        parameter_codec, sections, count_logits, gt_sections, gt_count,
        sequence_type=grassdata.SEQUENCE_TYPE_WING):
    if parameter_codec.sequence_type != sequence_type:
        raise ValueError(
            f'Wing loss received {parameter_codec.sequence_type!r} parameter codec.'
        )
    sections = parameter_codec.normalize(sections, gt_count)
    gt_sections = parameter_codec.normalize(gt_sections, gt_count)
    mask = grassdata.section_mask(
        gt_count, device=sections.device, sequence_type=sequence_type
    ).to(dtype=sections.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)

    def masked_field_mse(start, end):
        squared = (sections[..., start:end] - gt_sections[..., start:end]) ** 2
        per_section = squared.mean(dim=2)
        return (per_section * mask).sum(dim=1) / denominator

    cst_end = util.CST_AIRFOIL_CODE_SIZE
    position_end = cst_end + 3
    chord_end = position_end + 1
    cst_l = masked_field_mse(0, cst_end)
    position_l = masked_field_mse(cst_end, position_end)
    chord_l = masked_field_mse(position_end, chord_end)
    twist_l = masked_field_mse(chord_end, chord_end + 1)

    count_target = gt_count.reshape(-1) - grassdata.sequence_section_count_range(sequence_type)[0]
    count_l = F.cross_entropy(count_logits, count_target, reduction='none')
    total_l = (
        util.WING_LOSS_WEIGHTS['position'] * position_l
        + util.WING_LOSS_WEIGHTS['chord'] * chord_l
        + util.WING_LOSS_WEIGHTS['twist'] * twist_l
        + util.WING_LOSS_WEIGHTS['cst_code'] * cst_l
        + util.WING_LOSS_WEIGHTS['section_count'] * count_l
    )
    return {
        'position': position_l,
        'chord': chord_l,
        'twist': twist_l,
        'cst_code': cst_l,
        'section_count': count_l,
        'total': total_l,
    }


def fuselage_section_reconstruction_losses(
        parameter_codec, sections, count_logits, gt_sections, gt_count):
    if parameter_codec.sequence_type != grassdata.SEQUENCE_TYPE_FUSELAGE:
        raise ValueError(
            f'Fuselage loss received {parameter_codec.sequence_type!r} parameter codec.'
        )
    sections = parameter_codec.normalize(sections, gt_count)
    gt_sections = parameter_codec.normalize(gt_sections, gt_count)
    mask = grassdata.section_mask(
        gt_count, device=sections.device, sequence_type=grassdata.SEQUENCE_TYPE_FUSELAGE
    ).to(dtype=sections.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    position_l = (
        ((sections[..., :3] - gt_sections[..., :3]) ** 2).mean(dim=2) * mask
    ).sum(dim=1) / denominator
    size_l = (
        ((sections[..., 3:5] - gt_sections[..., 3:5]) ** 2).mean(dim=2) * mask
    ).sum(dim=1) / denominator
    count_target = gt_count.reshape(-1) - grassdata.sequence_section_count_range(
        grassdata.SEQUENCE_TYPE_FUSELAGE
    )[0]
    count_l = F.cross_entropy(count_logits, count_target, reduction='none')
    total_l = (
        util.FUSELAGE_LOSS_WEIGHTS['position'] * position_l
        + util.FUSELAGE_LOSS_WEIGHTS['size'] * size_l
        + util.FUSELAGE_LOSS_WEIGHTS['section_count'] * count_l
    )
    return {
        'position': position_l,
        'size': size_l,
        'section_count': count_l,
        'total': total_l,
    }


class GRASSDecoder(nn.Module):
    def __init__(self, config, section_statistics):
        super(GRASSDecoder, self).__init__()
        self.box_decoder = BoxDecoder(feature_size = config.feature_size, box_size = config.box_code_size)
        self.obb_box_decoder = TypedBoxDecoder(
            feature_size = config.feature_size,
            output_size = util.OBB_GEOMETRY_SIZE,
            tanh_size = util.OBB_GEOMETRY_SIZE,
        )
        self.fuselage_section_decoder = AutoregressiveSectionDecoder(
            grassdata.SEQUENCE_TYPE_FUSELAGE,
            config.feature_size,
            config.ae_rnn_type,
            section_statistics[grassdata.SEQUENCE_TYPE_FUSELAGE],
        )
        self.wing_section_decoder = AutoregressiveSectionDecoder(
            grassdata.SEQUENCE_TYPE_WING,
            config.feature_size,
            config.ae_rnn_type,
            section_statistics[grassdata.SEQUENCE_TYPE_WING],
        )
        self.adj_decoder = AdjDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.sym_decoder = SymDecoder(feature_size = config.feature_size, symmetry_size = config.symmetry_size, hidden_size = config.hidden_size)
        self.sample_decoder = SampleDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_classifier = NodeClassifier(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.component_classifier = ComponentClassifier(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.mseLoss = nn.MSELoss()  # pytorch's mean squared error loss
        self.creLoss = nn.CrossEntropyLoss()  # pytorch's cross entropy loss (NOTE: no softmax is needed before)

    def boxDecoder(self, feature):
        return self.box_decoder(feature)

    def obbBoxDecoder(self, feature):
        return self.obb_box_decoder(feature)

    def fuselageSectionDecoder(
            self, feature, teacher_sections, teacher_count, teacher_forcing_probability):
        return self.fuselage_section_decoder(
            feature, teacher_sections, teacher_count, teacher_forcing_probability
        )

    def wingSectionDecoder(
            self, feature, teacher_sections, teacher_count, teacher_forcing_probability):
        return self.wing_section_decoder(
            feature, teacher_sections, teacher_count, teacher_forcing_probability
        )

    def generateFuselageSections(self, feature):
        return self.fuselage_section_decoder.generate(feature)

    def generateWingSections(self, feature):
        return self.wing_section_decoder.generate(feature)

    def adjDecoder(self, feature):
        return self.adj_decoder(feature)

    def symDecoder(self, feature):
        return self.sym_decoder(feature)

    def sampleDecoder(self, feature):
        return self.sample_decoder(feature)

    def nodeClassifier(self, feature):
        return self.node_classifier(feature)

    def componentClassifier(self, feature):
        return self.component_classifier(feature)

    def decode_free(self, root_features):
        """Expand each root by its predicted node types within global safety limits."""
        if root_features.dim() != 2:
            raise ValueError('root_features must have shape [B, feature_size]')
        return [
            self._decode_free_sample(root_features[index:index + 1])
            for index in range(root_features.size(0))
        ]

    def _decode_free_sample(self, root_feature):
        max_leaves = util.FREE_DECODE_MAX_LEAF_NODES
        max_depth = util.FREE_DECODE_MAX_TREE_DEPTH

        def decode_node(feature, depth, leaf_capacity):
            node_logits = self.node_classifier(feature)
            allowed = torch.ones(3, dtype=torch.bool, device=feature.device)
            if leaf_capacity < 2:
                allowed[Tree.NodeType.ADJ.value] = False
            if depth >= max_depth:
                allowed[:] = False
                allowed[Tree.NodeType.BOX.value] = True
            constrained_logits = node_logits.masked_fill(~allowed.unsqueeze(0), float('-inf'))
            predicted_type = int(torch.argmax(node_logits, dim=1).item())
            node_type = int(torch.argmax(constrained_logits, dim=1).item())
            forced = predicted_type != node_type
            common = {
                'node_type': node_type,
                'node_logits': node_logits,
                'forced_by_limit': forced,
            }
            if node_type == Tree.NodeType.BOX.value:
                component_logits = self.component_classifier(feature)
                component = int(torch.argmax(component_logits, dim=1).item())
                if component == util.COMPONENT_FUSELAGE:
                    generated = self.fuselage_section_decoder.generate(feature)
                elif component == util.COMPONENT_WING:
                    generated = self.wing_section_decoder.generate(feature)
                else:
                    raise RuntimeError(f'free decoder predicted unsupported component {component}')
                sections, section_count, valid_mask, count_logits = generated
                common.update({
                    'component': component,
                    'component_logits': component_logits,
                    'sections': sections,
                    'section_count': section_count,
                    'valid_mask': valid_mask,
                    'count_logits': count_logits,
                })
                return common, 1, depth, int(forced)
            if node_type == Tree.NodeType.ADJ.value:
                left_feature, right_feature = self.adj_decoder(feature)
                left, left_leaves, left_depth, left_forced = decode_node(
                    left_feature, depth + 1, leaf_capacity - 1
                )
                right, right_leaves, right_depth, right_forced = decode_node(
                    right_feature, depth + 1, leaf_capacity - left_leaves
                )
                common.update({'left': left, 'right': right})
                return (
                    common,
                    left_leaves + right_leaves,
                    max(left_depth, right_depth),
                    int(forced) + left_forced + right_forced,
                )
            if node_type == Tree.NodeType.SYM.value:
                generator_feature, symmetry = self.sym_decoder(feature)
                generator, leaves, child_depth, child_forced = decode_node(
                    generator_feature, depth + 1, leaf_capacity
                )
                common.update({'generator': generator, 'symmetry': symmetry})
                return common, leaves, child_depth, int(forced) + child_forced
            raise RuntimeError(f'node classifier produced unsupported node type {node_type}')

        root, leaf_count, tree_depth, forced_count = decode_node(root_feature, 1, max_leaves)
        return {
            'root': root,
            'tree_valid': True,
            'leaf_count': leaf_count,
            'tree_depth': tree_depth,
            'forced_by_limit_count': forced_count,
        }

    def boxLossEstimator(self, box_feature, gt_box_feature):
        import torch
        losses = []
        for b, gt in zip(box_feature, gt_box_feature):
            geom_l = self.mseLoss(b[:10], gt[:10])

            # creLoss expects input (1, C) and target (1)
            pred_logits = b[10:].unsqueeze(0)
            target_class = torch.argmax(gt[10:]).unsqueeze(0)
            cls_l = self.creLoss(pred_logits, target_class)

            zero = torch.zeros_like(geom_l)
            losses.append(torch.stack([geom_l, zero, zero, zero, zero, zero, cls_l]))

        return torch.stack(losses, 0)

    def obbBoxLossEstimator(self, geometry, component_logits, gt_geometry, gt_component):
        payload_l = torch.mean((geometry - gt_geometry) ** 2, dim=1)
        component_l = F.cross_entropy(component_logits, gt_component.reshape(-1), reduction='none')
        zero = torch.zeros_like(payload_l)
        return torch.stack([payload_l, zero, zero, zero, zero, zero, component_l], dim=1)

    def wingSectionLossEstimator(
            self, sections, count_logits, component_logits, gt_sections, gt_count, gt_component,
            sequence_type=grassdata.SEQUENCE_TYPE_WING):
        reconstruction_losses = wing_section_reconstruction_losses(
            self.wing_section_decoder.parameter_codec,
            sections,
            count_logits,
            gt_sections,
            gt_count,
            sequence_type,
        )
        component_l = F.cross_entropy(component_logits, gt_component.reshape(-1), reduction='none')
        zero = torch.zeros_like(reconstruction_losses['total'])
        return torch.stack(
            [reconstruction_losses['total'], zero, zero, zero, zero, zero, component_l], dim=1
        )

    def fuselageSectionLossEstimator(
            self, sections, count_logits, component_logits, gt_sections, gt_count, gt_component):
        reconstruction_losses = fuselage_section_reconstruction_losses(
            self.fuselage_section_decoder.parameter_codec,
            sections,
            count_logits,
            gt_sections,
            gt_count,
        )
        component_l = F.cross_entropy(component_logits, gt_component.reshape(-1), reduction='none')
        zero = torch.zeros_like(reconstruction_losses['total'])
        return torch.stack(
            [reconstruction_losses['total'], zero, zero, zero, zero, zero, component_l], dim=1
        )

    def symLossEstimator(self, sym_param, gt_sym_param):
        import torch
        return torch.stack([self.mseLoss(s, gt) for s, gt in zip(sym_param, gt_sym_param)], 0)

    def classifyLossEstimator(self, label_vector, gt_label_vector):
        import torch
        return torch.stack([self.creLoss(l.unsqueeze(0), gt.unsqueeze(0)) for l, gt in zip(label_vector, gt_label_vector)], 0)


def decode_structure_fold(
        fold, feature, tree, use_sample_decoder=True, teacher_forcing_probability=None):
    box_losses = []
    sym_losses = []
    cat_losses = []

    def decode_node_box(node, feature):
        if node.is_leaf():
            if isinstance(node.box, dict):
                if teacher_forcing_probability is None:
                    raise ValueError('teacher_forcing_probability is required for sequence reconstruction.')
                component = node.box["component"]
                sequence_type = node.box['sequence_type']
                component_logits = fold.add('componentClassifier', feature)
                component_target = torch.LongTensor([component])
                if sequence_type == grassdata.SEQUENCE_TYPE_WING:
                    sections, count_logits = fold.add(
                        'wingSectionDecoder', feature, node.box["sections"],
                        node.box["section_count"], teacher_forcing_probability
                    ).split(2)
                    box_losses.append(fold.add(
                        'wingSectionLossEstimator',
                        sections,
                        count_logits,
                        component_logits,
                        node.box["sections"],
                        node.box["section_count"],
                        component_target,
                    ))
                elif sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
                    sections, count_logits = fold.add(
                        'fuselageSectionDecoder', feature, node.box["sections"],
                        node.box["section_count"], teacher_forcing_probability
                    ).split(2)
                    box_losses.append(fold.add(
                        'fuselageSectionLossEstimator',
                        sections,
                        count_logits,
                        component_logits,
                        node.box["sections"],
                        node.box["section_count"],
                        component_target,
                    ))
                elif component == util.COMPONENT_ENGINE:
                    box = fold.add('obbBoxDecoder', feature)
                    box_losses.append(fold.add(
                        'obbBoxLossEstimator',
                        box,
                        component_logits,
                        node.box["geometry"],
                        component_target,
                    ))
                else:
                    raise ValueError(f"Unknown component type: {component}")
            else:
                box = fold.add('boxDecoder', feature)
                box_losses.append(fold.add('boxLossEstimator', box, node.box))
            label = fold.add('nodeClassifier', feature)
            cat_losses.append(fold.add('classifyLossEstimator', label, node.label))
        elif node.is_adj():
            left, right = fold.add('adjDecoder', feature).split(2)
            decode_node_box(node.left, left)
            decode_node_box(node.right, right)
            label = fold.add('nodeClassifier', feature)
            cat_losses.append(fold.add('classifyLossEstimator', label, node.label))
        elif node.is_sym():
            sym_gen, sym_param = fold.add('symDecoder', feature).split(2)
            sym_losses.append(fold.add('symLossEstimator', sym_param, node.sym))
            decode_node_box(node.left, sym_gen)
            label = fold.add('nodeClassifier', feature)
            cat_losses.append(fold.add('classifyLossEstimator', label, node.label))

    if use_sample_decoder:
        feature = fold.add('sampleDecoder', feature)
    decode_node_box(tree.root, feature)
    return box_losses, sym_losses, cat_losses


#########################################################################################
## Functions for model testing: Decode a root code into a tree structure of boxes
#########################################################################################

def vrrotvec2mat(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    m = torch.tensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]], device=rotvector.device)
    return m

def decode_structure(model, root_code):
    """
    Decode a root code into a tree structure of boxes
    """
    decode = model.sampleDecoder(root_code)
    syms = [torch.ones(8, device=root_code.device).mul(10)]
    #初始化工作栈（放入根节点），以及一个空列表用于收集最终生成的所有 3D 包围盒。
    stack = [decode]
    boxes = []
    #只要栈里还有未处理的节点，就持续循环
    while len(stack) > 0:
        f = stack.pop() #取出当前需要解析的节点特征向量 f
        label_prob = model.nodeClassifier(f) #利用分类器网络预测该特征究竟属于哪种节点
        _, label = torch.max(label_prob, 1) #取概率最高的值
        label = label.detach()
        if label[0] == 1:  # ADJ
            left, right = model.adjDecoder(f)
            stack.append(left)
            stack.append(right)
            s = syms.pop() # 向下传递对称属性。弹出一个当前的对称状态，再复制两份压回去，确保左右子节点都能继承到相同的对称指令
            syms.append(s)
            syms.append(s)
        if label[0] == 2:  # SYM
            left, s = model.symDecoder(f)
            s = s.squeeze(0)
            stack.append(left)
            syms.pop() #把旧的对称状态扔掉，将模型刚刚预测出的有效对称参数压入栈。这样它下方的所有叶子节点就能拿到这组参数进行阵列
            syms.append(s.detach())
        if label[0] == 0:  # BOX
            reBox = model.boxDecoder(f) #解码OBB
            reBoxes = [reBox]
            s = syms.pop() #提取当前叶子节点继承到的对称参数
            l1 = abs(s[0] + 1)
            l2 = abs(s[0])
            l3 = abs(s[0] - 1)

            if l1 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.detach().squeeze(0), 1, 0)
                f1 = torch.cat([sList[1], sList[2], sList[3]])
                f1 = f1/torch.norm(f1)
                f2 = torch.cat([sList[4], sList[5], sList[6]])
                folds = round((1/s[7]).item())
                for i in range(folds-1):
                    rotvector = torch.cat([f1, sList[7].mul(2*3.1415).mul(i+1)])
                    rotm = vrrotvec2mat(rotvector)
                    c1 = torch.cat([bList[0], bList[1], bList[2]])
                    c2 = torch.cat([bList[3], bList[4], bList[5]])
                    dims = torch.cat([bList[6], bList[7], bList[8], bList[9]])
                    
                    newc1 = rotm.matmul(c1.add(-f2)).add(f2)
                    newc2 = rotm.matmul(c2.add(-f2)).add(f2)
                    
                    if len(bList) >= 13:
                        cls = torch.cat([bList[10], bList[11], bList[12]])
                        newbox = torch.cat([newc1, newc2, dims, cls])
                    else:
                        newbox = torch.cat([newc1, newc2, dims])
                    reBoxes.append(newbox.unsqueeze(0))

            if l2 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.detach().squeeze(0), 1, 0)
                trans = torch.cat([sList[1], sList[2], sList[3]])
                trans_end = torch.cat([sList[4], sList[5], sList[6]])
                c1 = torch.cat([bList[0], bList[1], bList[2]])
                c2 = torch.cat([bList[3], bList[4], bList[5]])
                dims = torch.cat([bList[6], bList[7], bList[8], bList[9]])
                
                trans_length = math.sqrt(torch.sum(trans**2).item())
                # Use c1 as reference point for translational total length
                trans_total = math.sqrt(torch.sum(trans_end.add(-c1)**2).item())
                folds = round(trans_total/max(trans_length, 1e-6))
                for i in range(folds):
                    c1 = torch.cat([bList[0], bList[1], bList[2]])
                    c2 = torch.cat([bList[3], bList[4], bList[5]])
                    newc1 = c1.add(trans.mul(i+1))
                    newc2 = c2.add(trans.mul(i+1))
                    if len(bList) >= 13:
                        cls = torch.cat([bList[10], bList[11], bList[12]])
                        newbox = torch.cat([newc1, newc2, dims, cls])
                    else:
                        newbox = torch.cat([newc1, newc2, dims])
                    reBoxes.append(newbox.unsqueeze(0))

            if l3 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.detach().squeeze(0), 1, 0)
                ref_normal = torch.cat([sList[1], sList[2], sList[3]])
                ref_normal = ref_normal/torch.norm(ref_normal)
                ref_point = torch.cat([sList[4], sList[5], sList[6]])
                
                c1 = torch.cat([bList[0], bList[1], bList[2]])
                c2 = torch.cat([bList[3], bList[4], bList[5]])
                dims = torch.cat([bList[6], bList[7], bList[8], bList[9]])
                
                # Reflection logic for points c1 and c2
                v1 = c1.add(-ref_point)
                dist1 = torch.sum(v1 * ref_normal)
                newc1 = c1.add(ref_normal.mul(-2 * dist1))
                
                v2 = c2.add(-ref_point)
                dist2 = torch.sum(v2 * ref_normal)
                newc2 = c2.add(ref_normal.mul(-2 * dist2))
                
                if len(bList) >= 13:
                    cls = torch.cat([bList[10], bList[11], bList[12]])
                    newbox = torch.cat([newc1, newc2, dims, cls])
                else:
                    newbox = torch.cat([newc1, newc2, dims])
                reBoxes.append(newbox.unsqueeze(0))

            boxes.extend(reBoxes)

    return boxes
