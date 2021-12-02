from transformers.models.bert.modeling_bert import BertModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.mt5.modeling_mt5 import MT5EncoderModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import util
import logging
from transformers import PretrainedConfig
from torch.nn.utils.rnn import pack_padded_sequence
from modeling_graph import TransformerRootPathLayer, TransformerEncoder, set_custom_config

logger = logging.getLogger(__name__)


def get_seq_encoder(config):
    if config['model_type'] == 'bert':
        return BertModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlm-roberta':
        return XLMRobertaModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'mt5':
        return MT5EncoderModel.from_pretrained(config['pretrained'])
    else:
        raise ValueError(config['model_type'])


class TransformerQaGraph(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.seq_encoder = get_seq_encoder(config)
        self.seq_config = self.seq_encoder.config

        self.seq_hidden_size = self.seq_config.hidden_size
        self.seq_feat_hidden_size = self.seq_hidden_size + sum([config['use_pos_feature'], config['use_depth_feature'],
                                                                config['node_rel_feature']]) * config['feat_emb_size']
        self.seq_feat_head_size = self.seq_feat_hidden_size // self.seq_config.num_attention_heads
        self.path_hidden_size = config['path_hidden'] if config['path_hidden'] > 1 else \
            int(config['path_hidden'] * self.seq_feat_hidden_size)
        set_custom_config(config)

        self.emb_pos = nn.Embedding(len(util.upos_to_id), config['feat_emb_size'])
        self.emb_depth = nn.Embedding(config['max_node_depth'] + 1, config['feat_emb_size'])
        self.emb_rel = nn.Embedding(len(util.rel_to_id), config['feat_emb_size'])

        self.emb_graph_rel = nn.Embedding(len(util.rel_to_id), self.seq_feat_head_size) if config['use_graph_rel'] else None
        self.emb_path_rel = nn.Embedding(len(util.rel_to_id), self.seq_feat_hidden_size) if config['use_root_path'] else None

        self.graph_rel_config = self.create_graph_rel_config()
        self.path_config = self.create_path_config()
        self.final_config = self.create_final_config()

        self.dropout = nn.Dropout(self.graph_rel_config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.seq_feat_hidden_size, eps=self.graph_rel_config.layer_norm_eps)

        self.graph_rel_encoder = None
        if config['use_graph_rel']:
            self.graph_rel_encoder = TransformerEncoder(self.graph_rel_config)

        self.path_seq_encoder, self.path_encoder = None, None
        if config['use_root_path']:
            if config['use_undirected_path']:
                self.path_seq_encoder = nn.LSTM(self.seq_feat_hidden_size, self.path_hidden_size // 2, bidirectional=True)
            else:
                self.path_seq_encoder = nn.LSTM(self.seq_feat_hidden_size, self.path_hidden_size, bidirectional=True)
            self.path_encoder = TransformerRootPathLayer(self.path_config)

        self.final_encoder = None
        if config['final_layer']:
            self.final_encoder = TransformerEncoder(self.final_config)

        self.tag_outputs = None
        self.qa_outputs = nn.Linear(self.final_config.hidden_size, 2)

    def _create_config_template(self):
        return PretrainedConfig.from_pretrained('bert-base-multilingual-cased' if self.seq_config.hidden_size == 768
                                                else 'xlm-roberta-large')

    def create_graph_rel_config(self):
        config = self._create_config_template()
        config.hidden_size = self.seq_feat_hidden_size
        config.intermediate_size = config.hidden_size * 4
        config.num_hidden_layers = self.config['graph_layer']
        return config

    def create_path_config(self):
        config = self._create_config_template()
        config.hidden_size = self.path_hidden_size
        config.intermediate_size = config.hidden_size * 4
        config.num_attention_heads = self.config['path_attention_heads']
        config.num_hidden_layers = self.config['graph_layer']
        return config

    def create_final_config(self):
        config = self._create_config_template()
        config.hidden_size = self.seq_feat_hidden_size
        if self.config['use_root_path']:
            config.hidden_size += self.path_hidden_size
        config.intermediate_size = config.hidden_size * 4
        if self.config['use_root_path']:
            config.num_attention_heads += self.path_config.num_attention_heads
        config.num_hidden_layers = self.config['final_layer']
        return config

    @classmethod
    def extend_tensor(cls, tensor):
        # Make it broadcastable for heads and tokens
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1).unsqueeze(1)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        return tensor

    def get_tagging_loss(self, logits, attention_mask, start_positions, end_positions):
        batch_size, seq_len, device = logits.shape[0], logits.shape[1], logits.device
        indices = torch.arange(0, seq_len, device=logits.device).unsqueeze(0)
        start_positions = start_positions.unsqueeze(1)
        end_positions = end_positions.unsqueeze(1)
        pos_mask = (indices >= start_positions) & (indices <= end_positions)
        neg_mask = torch.logical_not(pos_mask) & attention_mask.to(torch.bool)
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        num_pos = min(15 * batch_size, pos_logits.shape[0])
        num_neg = min(2 * num_pos, neg_logits.shape[0])
        pos_logits = util.random_select(pos_logits, num_pos)
        neg_logits = util.random_select(neg_logits, num_neg)

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(torch.cat([pos_logits, neg_logits]),
                        torch.cat([torch.ones(num_pos, dtype=torch.float, device=device),
                                   torch.zeros(num_neg, dtype=torch.float, device=device)]))
        return loss

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                start_positions=None, end_positions=None, pos_ids=None, depths=None, rels=None,
                graph_rels=None, from_paths=None, path_lens=None):
        conf, batch_size, seq_len = self.config, input_ids.shape[0], input_ids.shape[1]
        # Get extended attention mask
        extended_attention_mask = self.extend_tensor(attention_mask)  # [batch size, 1, 1, seq len]
        extended_attention_mask = (1 - extended_attention_mask.to(torch.float)) * -10000
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                  'output_attentions': False, 'output_hidden_states': False, 'return_dict': False}
        if conf['model_type'] != 'mt5':
            inputs['token_type_ids'] = token_type_ids
        outputs = self.seq_encoder(**inputs)
        sequence_output = outputs[0]  # [batch size, seq len, seq hidden]

        features = []
        # Add pos feature
        if conf['use_pos_feature']:
            pos_emb = self.emb_pos(pos_ids)
            features.append(pos_emb)
        # Add depth feature
        if conf['use_depth_feature']:
            depth_emb = self.emb_depth(depths)
            features.append(depth_emb)
        # Add rel feature
        if conf['node_rel_feature']:
            rel_emb = self.emb_rel(rels)
            features.append(rel_emb)
        sequence_output = torch.cat([sequence_output] + features, dim=-1)
        if features and False:
            sequence_output = self.LayerNorm(sequence_output)

        # Get graph rel output
        graph_rels_output = None
        if conf['use_graph_rel']:
            graph_rels = self.extend_tensor(graph_rels)
            graph_mask = (graph_rels > 0).to(torch.float)
            graph_rels = self.emb_graph_rel(graph_rels)
            graph_rels_output = self.graph_rel_encoder(sequence_output, extended_attention_mask,
                                                       graph_rels=graph_rels, graph_mask=graph_mask)[0]

        # Get path output
        path_output = None
        if conf['use_root_path']:
            # Get path encoding
            max_path_len = from_paths.shape[2]
            path_lens = path_lens.view(-1)
            node_edge_emb = torch.cat([sequence_output, self.emb_path_rel.weight.unsqueeze(0).repeat(batch_size, 1, 1)], dim=1)

            from_paths = from_paths.view(batch_size, -1)
            from_path_hidden = util.batch_select(node_edge_emb, from_paths)
            from_path_hidden = from_path_hidden.view(batch_size * seq_len, max_path_len, -1)
            from_path_hidden = pack_padded_sequence(from_path_hidden.transpose(0, 1), path_lens.cpu(), enforce_sorted=False)

            _, (path_hidden, _) = self.path_seq_encoder(from_path_hidden)
            if conf['use_undirected_path']:
                from_path_hidden = path_hidden.transpose(0, 1).view(batch_size, seq_len, -1)
                to_path_hidden = from_path_hidden
            else:
                path_hidden = path_hidden.transpose(0, 1).view(batch_size, seq_len, 2, -1).transpose(-1, -2)
                from_path_hidden, to_path_hidden = path_hidden.split(1, -1)
                from_path_hidden, to_path_hidden = from_path_hidden.squeeze(-1), to_path_hidden.squeeze(-1)

            path_output = self.path_encoder(sequence_output, extended_attention_mask,
                                            from_path_hidden=from_path_hidden, to_path_hidden=to_path_hidden)[0]

        # Get final output
        if graph_rels_output is not None:
            sequence_output = 0.5 * sequence_output + 0.5 * graph_rels_output
        if path_output is not None:
            sequence_output = torch.cat([sequence_output, path_output], dim=-1)
        if conf['final_layer']:
            sequence_output = self.final_encoder(sequence_output, extended_attention_mask)[0]

        # Get tagging logits
        tagging_logits = None
        if self.tag_outputs is not None:
            tagging_logits = self.tag_outputs(sequence_output).squeeze(-1)

        # Get QA logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        # Get loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            if tagging_logits is not None:
                tagging_loss = self.get_tagging_loss(tagging_logits, attention_mask, start_positions, end_positions)
                total_loss = (start_loss + end_loss) / 2 + tagging_loss * conf['tagging_loss_coef']
            else:
                total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) if tagging_logits is None else (start_logits, end_logits, tagging_logits)
        return (total_loss, output) if total_loss is not None else output
