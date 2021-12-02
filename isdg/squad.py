import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from transformers.tokenization_utils_base import TruncationStrategy
from util import unicode_whitespace, upos_to_id, rel_to_id
from visualize import visualize_root_path, visualize_graph_rel
from os.path import join
import gc


MULTI_SEP_TOKENS_TOKENIZERS_SET = {"xlmroberta", "roberta", "camembert", "bart"}  # Why roberta??

logger = logging.getLogger(__name__)


class QaDataset(Dataset):
    def __init__(self, config, features, is_training):
        super().__init__()
        self.initialized = False
        self.is_training = is_training

        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        self.all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        self.all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        self.all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        self.all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        self.all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        self.all_graph_indices = [f.seq_graph_indices for f in features]
        self.all_graph_rel_ids = [f.seq_graph_rel_ids for f in features]

        self.all_pos = torch.tensor([f.pos for f in features], dtype=torch.long)
        self.all_node_depth = torch.tensor([f.node_depth for f in features], dtype=torch.long)
        self.all_node_rel = torch.tensor([f.node_rel for f in features], dtype=torch.long)

        # Build paths
        self.all_from_paths, self.all_path_lens = None, None
        if features[0].seq_node_paths:
            self.all_from_paths, self.all_path_lens = [], []
            max_path_len, max_seq_len = config['max_node_depth'] * 2 - 1, config['max_segment_len']
            for f in features:
                node_paths, edge_paths = f.seq_node_paths, f.seq_edge_paths
                paths, path_lens = [], []
                for node_path, edge_path in zip(node_paths, edge_paths):
                    path = [node_path[0]]
                    for node_i, rel_id in zip(node_path[1:], edge_path):
                        path.append(rel_id + max_seq_len)  # Distinguish node and edge
                        path.append(node_i)
                    path_lens.append(len(path))
                    paths.append(path + [0] * (max_path_len - len(path)))
                self.all_from_paths.append(paths)
                self.all_path_lens.append(path_lens)
            self.all_from_paths = torch.tensor(self.all_from_paths, dtype=torch.long)
            self.all_path_lens = torch.tensor(self.all_path_lens, dtype=torch.long)

        # For eval
        self.all_feature_idx = torch.arange(len(features), dtype=torch.long)  # To identify which feature in eval
        # For training
        self.all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        self.all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)

    def initialize(self, config):
        if not config['use_root_path']:
            del self.all_from_paths
            del self.all_path_lens
            self.all_from_paths = self.all_path_lens = None
        if not config['use_graph_rel']:
            del self.all_graph_indices
            del self.all_graph_rel_ids
            self.all_graph_indices = self.all_graph_rel_ids = None
        self.initialized = True
        # gc.collect()

    def __getitem__(self, i):
        assert self.initialized, 'Dataset should be initialized first'
        # Return instance based on for training or eval
        if not self.is_training:
            to_return = self.all_input_ids[i], self.all_attention_masks[i], self.all_token_type_ids[i], \
                        self.all_feature_idx[i], self.all_cls_index[i], self.all_p_mask[i],
        else:
            to_return = self.all_input_ids[i], self.all_attention_masks[i], self.all_token_type_ids[i], \
                        self.all_start_positions[i], self.all_end_positions[i], \
                        self.all_cls_index[i], self.all_p_mask[i], self.all_is_impossible[i]

        # For paths
        if self.all_from_paths is not None:
            to_return += (self.all_from_paths[i], self.all_path_lens[i])

        # For graph rels; generate graph rels on the fly
        graph_indices, graph_rel_ids, graph = self.all_graph_indices[i], self.all_graph_rel_ids[i], None
        if graph_indices:
            seq_indices, seq_rels = [], []
            for idx in range(len(graph_indices)):
                for out_idx, rel in zip(graph_indices[idx], graph_rel_ids[idx]):
                    seq_indices.append([idx, out_idx])
                    seq_rels.append(rel)
            graph = torch.sparse.LongTensor(torch.LongTensor(seq_indices).t(), torch.LongTensor(seq_rels),
                                            torch.Size([len(graph_indices), len(graph_indices)])).to_dense()
            to_return += (graph,)

        to_return += (self.all_pos[i], self.all_node_depth[i], self.all_node_rel[i])
        return to_return

    def __len__(self):
        return self.all_input_ids.shape[0]


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    # e.g. orig_answer_text = '***' but last original token is '***.'
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token; a token can appear in multiple overlapping segments"""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["paragraph_len"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["paragraph_len"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == '\u3000':
        return True
    if c in unicode_whitespace:
        return True
    return False


def get_pos_ids(pos_list):
    return [upos_to_id[tag] for tag in pos_list]


def adjust_rel(rel):
    # Do not use subrels to avoid sparsity
    if ':' in rel:
        rel = rel[:rel.find(':')]
    if rel not in rel_to_id:
        rel = 'punct'  # Use as default (most common type)
    return rel, f'R-{rel}'


def tokenize_subwords(tokens, tokenizer, throw_sentence_piece_space=False):
    # Subword tokenization on tokenized words; build map between subtokens and original tokens
    if throw_sentence_piece_space:
        unk_id = 3 if config['model_type'] == 'xlm-roberta' else 2 if config['model_type'] == 'mt5' else 100
        tok_to_orig_index, orig_to_tok_index = [], []
        all_subtokens = []
        for (i, token) in enumerate(tokens):
            orig_to_tok_index.append(len(all_subtokens))
            if tokenizer.convert_tokens_to_ids(token) == unk_id:  # Not in vocab
                sub_tokens = tokenizer.tokenize(token)
                # XLM-R and MT5 uses sentence-piece tokenization
                if config['model_type'] == 'xlm-roberta' and len(sub_tokens) > 1 and \
                        tokenizer.convert_tokens_to_ids(sub_tokens[0]) == 6:  # Throw out the added trailing space
                    sub_tokens = sub_tokens[1:]  # about same performance as without throwing out space
            else:
                sub_tokens = [token]  # Avoid adding trailing space to a different subtoken
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_subtokens.append(sub_token)
        return all_subtokens, tok_to_orig_index, orig_to_tok_index
    else:
        tok_to_orig_index, orig_to_tok_index = [], []
        all_subtokens = []
        for (i, token) in enumerate(tokens):
            orig_to_tok_index.append(len(all_subtokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_subtokens.append(sub_token)
        return all_subtokens, tok_to_orig_index, orig_to_tok_index


def adapt_to_sub(orig_l, tok_to_orig_index):
    if orig_l is None:
        return None
    sub_l = [orig_l[orig_i] for orig_i in tok_to_orig_index]
    return sub_l


def get_sub_root_indices(heads, tok_to_orig_index):
    sub_heads = adapt_to_sub(heads, tok_to_orig_index)
    return [h_i for h_i, h in enumerate(sub_heads) if h == -1]


def transform_dep_to_graph(heads, rels, sent_ids, subtokens, tok_to_orig_idx, orig_to_tok_idx):
    """ For use_graph_path or use_graph_rel: get graph on subword level """
    def set_graph(graph, node_depth, heads, sent_ids, root_idx):
        """ Set direct relations (parent and child)on the sentence subgraph """
        sent_start = sent_ids.index(sent_ids[root_idx])
        sent_end = sent_start + 1  # Exclusive
        while sent_end < len(sent_ids) and sent_ids[sent_end] == sent_ids[root_idx]:
            sent_end += 1

        # Build top-down subgraph for this sentence
        subgraph = defaultdict(lambda: set())
        for idx in range(sent_start, sent_end):
            if idx != root_idx:
                subgraph[heads[idx]].add(idx)

        # BFS to set node depth
        queue, current_depth = [root_idx], 0
        while queue:
            new_queue = []
            for idx in queue:
                node_depth[idx] = current_depth
                new_queue += [to_visit for to_visit in subgraph[idx]]
            if current_depth < config['max_node_depth']:
                current_depth += 1
            queue = new_queue

        # Set direct relation (parent and child)
        for idx in range(sent_start, sent_end):
            if idx == root_idx:
                continue
            head = heads[idx]
            assert head not in graph[idx], f'parent {head} should not have been added for {idx}'
            assert idx not in graph[head], f'child {idx} should not have been added for {head}'
            rel, r_rel = adjust_rel(rels[idx])
            graph[idx][head] = rel  # Set parent relation
            graph[head][idx] = r_rel  # Set child relation

    # Get graph and depth per sentence based on config
    graph = [dict() for _ in range(len(heads))]
    node_depth = [-1] * len(heads)
    sent_root_idx = [i for i, head in enumerate(heads) if head < 0]
    for root_idx in sent_root_idx:
        set_graph(graph, node_depth, heads, sent_ids, root_idx)
    # Check sanity
    assert all([d >= 0 for d in node_depth])
    assert all([node_depth[i] == 0 for i in sent_root_idx])  # Root has depth 0

    # Adapt subtoken depth
    subtoken_depth = [node_depth[tok_to_orig_idx[i]] for i in range(len(subtokens))]  # Root subtokens have depth 0

    # Set edges across sentences
    for root_idx in sent_root_idx:
        graph[root_idx].update({idx: 'cross-sent' for idx in sent_root_idx if idx != root_idx})
    # Check sanity
    for idx in range(len(heads)):
        assert idx not in graph[idx], f'self {idx} should not have been added'

    # Adapt dep graph to subtoken level
    orig_to_tok_indices = []
    for idx in range(len(heads)):
        subtoken_idx = orig_to_tok_idx[idx]
        subtoken_idx_till = orig_to_tok_idx[idx + 1] if idx < len(heads) - 1 else len(subtokens)  # Exclusive
        orig_to_tok_indices.append(set(range(subtoken_idx, subtoken_idx_till)))

    subtoken_graph = [dict() for _ in range(len(subtokens))]
    sub_idx = 0
    while sub_idx < len(subtokens):
        orig_idx = tok_to_orig_idx[sub_idx]

        out_edges_sub = {}
        for orig_out_idx, rel in graph[orig_idx].items():
            if orig_out_idx != orig_idx:
                for sub_out_idx in orig_to_tok_indices[orig_out_idx]:
                    out_edges_sub[sub_out_idx] = rel

        sub_idx_till = sub_idx + 1  # Exclusive
        while sub_idx_till < len(subtokens) and tok_to_orig_idx[sub_idx_till] == orig_idx:
            sub_idx_till += 1
        # Add common out edges for subtokens of current orig token
        for i in range(sub_idx, sub_idx_till):
            assert len(subtoken_graph[i]) == 0, f'subtoken graph for {i} should not have been added any edges'
            subtoken_graph[i].update(out_edges_sub)
        # Add edges among subtokens of current orig token
        for i in range(sub_idx, sub_idx_till):
            for j in range(sub_idx, sub_idx_till):
                if i != j:
                    subtoken_graph[i][j] = 'subword'
                else:
                    subtoken_graph[i][i] = 'self'  # Add self connection
        sub_idx = sub_idx_till

    return subtoken_graph, subtoken_depth


def get_paths_on_graph(graph):
    """ For use_graph_path; get complete paths on the sub-level graph """
    if graph is None:
        return None, None
    num_tokens = len(graph)

    def bfs(start):
        # BFS to set all paths from start
        queue = [start]  # self will be added at 1st iteration
        while queue:
            new_queue = []
            for from_i in queue:
                for to_i, rel_id in graph[from_i]:
                    if not node_paths[to_i]:
                        node_paths[to_i] = node_paths[from_i] + [to_i]
                        edge_paths[to_i] = edge_paths[from_i] + [rel_id]
                        new_queue.append(to_i)
            queue = new_queue

    all_node_paths, all_edge_paths = [], []
    for idx in range(num_tokens):
        node_paths, edge_paths = [[] for _ in range(num_tokens)], [[] for _ in range(num_tokens)]
        bfs(idx)
        for to_idx in range(num_tokens):
            assert len(node_paths[to_idx]) == len(edge_paths[to_idx])
            assert len(node_paths[to_idx]) >= 1, 'All nodes should be connected'
        node_paths = [tuple(path) for path in node_paths]
        edge_paths = [tuple(path) for path in edge_paths]
        all_node_paths.append(tuple(node_paths))
        all_edge_paths.append(tuple(edge_paths))

    return tuple(all_node_paths), tuple(all_edge_paths)  # [max seq len, max seq len, path len]


def get_paths_to_root(heads, rels, tok_to_orig_idx, orig_to_tok_idx):
    """ For use_root_path; get paths to the root on subword level """
    def get_path(idx):
        node_path, edge_path = [idx], []
        while heads[idx] >= 0:
            node_path.append(heads[idx])
            edge_path.append(adjust_rel(rels[idx])[0])
            idx = heads[idx]
        assert rels[idx] in ('root', 'punct'), 'Path should end before root or punct (default)'
        # Truncate path
        node_path = node_path[:config['max_node_depth']]
        edge_path = edge_path[:len(node_path) - 1]
        return node_path, edge_path

    node_paths, edge_paths = [], []
    for i in range(len(heads)):
        node_path, edge_path = get_path(i)
        node_paths.append(node_path)
        edge_paths.append(edge_path)

    sub_node_paths, sub_edge_paths = [], []
    for i, orig_i in enumerate(tok_to_orig_idx):
        sub_node_path = [i] + [orig_to_tok_idx[head] for head in node_paths[orig_i][1:]]
        sub_node_paths.append(tuple(sub_node_path))
        sub_edge_paths.append(tuple(edge_paths[orig_i]))

    return tuple(sub_node_paths), tuple(sub_edge_paths)


def squad_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):
    features = []
    # Check sanity; if the answer cannot be found in the text, then skip this example.
    if is_training and not example.is_impossible:
        start_position = example.start_position
        end_position = example.end_position

        # Check with no-space text since tokenization is different
        actual_text = "".join(example.doc_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = "".join(example.answer_text.strip().split())
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    # Get subtokens
    context_subtokens, context_tok_to_orig_idx, context_orig_to_tok_idx = tokenize_subwords(example.doc_tokens, tokenizer)
    context_pos_ids = get_pos_ids(adapt_to_sub(example.pos_tags, context_tok_to_orig_idx))
    context_rels = [adjust_rel(rel)[0] for rel in adapt_to_sub(example.rels, context_tok_to_orig_idx)]
    context_root_indices = get_sub_root_indices(example.heads, context_tok_to_orig_idx)
    question_subtokens, question_tok_to_orig_idx, question_orig_to_tok_idx = tokenize_subwords(example.question_tokens, tokenizer)
    question_pos_ids = get_pos_ids(adapt_to_sub(example.question_pos_tags, question_tok_to_orig_idx))
    question_rels = [adjust_rel(rel)[0] for rel in adapt_to_sub(example.question_rels, question_tok_to_orig_idx)]
    question_root_indices = get_sub_root_indices(example.question_heads, question_tok_to_orig_idx)

    # Get answer positions for subtokens
    if is_training and not example.is_impossible:
        tok_start_position = context_orig_to_tok_idx[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = context_orig_to_tok_idx[example.end_position + 1] - 1
        else:
            tok_end_position = len(context_subtokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            context_subtokens, tok_start_position, tok_end_position, tokenizer, example.answer_text)

    assert example.heads, 'No dependency parsing for context'
    assert example.question_heads, 'No dependency parsing for question'

    # For use_graph_path or use_graph_rel; get graph
    context_graph, context_depth = transform_dep_to_graph(example.heads, example.rels, example.sent_ids,
                                                          context_subtokens, context_tok_to_orig_idx,
                                                          context_orig_to_tok_idx)
    question_graph, question_depth = transform_dep_to_graph(example.question_heads, example.question_rels,
                                                            example.question_sent_ids, question_subtokens,
                                                            question_tok_to_orig_idx, question_orig_to_tok_idx)

    # For use_root_path; get root paths
    context_node_paths, context_edge_paths, question_node_paths, question_edge_paths = None, None, None, None
    if config['use_root_path'] or True:
        context_node_paths, context_edge_paths = get_paths_to_root(example.heads, example.rels,
                                                                   context_tok_to_orig_idx, context_orig_to_tok_idx)
        question_node_paths, question_edge_paths = get_paths_to_root(example.question_heads, example.question_rels,
                                                                     question_tok_to_orig_idx, question_orig_to_tok_idx)

    # Truncate and encode query first
    if len(question_subtokens) > max_query_length:
        question_subtokens = question_subtokens[:max_query_length]
        question_tok_to_orig_idx = question_tok_to_orig_idx[:max_query_length]
        question_pos_ids = question_pos_ids[:max_query_length]
        question_rels = question_rels[:max_query_length]
        question_root_indices = [r_i for r_i in question_root_indices if r_i < max_query_length]
        question_depth = question_depth[:max_query_length]
        question_graph = question_graph[:max_query_length]
        for idx in range(len(question_graph)):
            question_graph[idx] = {out_idx: rel for out_idx, rel in question_graph[idx].items() if
                                   out_idx < max_query_length}
        if config['use_root_path'] or True:
            new_question_node_paths, new_question_edge_paths = [], []
            for node_path, edge_path in zip(question_node_paths[:max_query_length], question_edge_paths[:max_query_length]):
                new_node_path, new_edge_path = [node_path[0]], []
                for i, rel in zip(node_path[1:], edge_path):
                    if i < max_query_length:
                        new_node_path.append(i)
                        new_edge_path.append(rel)
                new_question_node_paths.append(tuple(new_node_path))
                new_question_edge_paths.append(tuple(new_edge_path))
            question_node_paths, question_edge_paths = tuple(new_question_node_paths), tuple(new_question_edge_paths)
    truncated_query = tokenizer.convert_tokens_to_ids(question_subtokens)

    # For use_graph_path; combine context and question graph for full graph path
    if config['use_graph_path']:
        combined_graph = None
        context_graph_tmp = [{i + len(question_graph): rel for i, rel in edges.items()} for edges in context_graph]
        combined_graph = question_graph + context_graph_tmp
        # Add cross-type edge between question and context
        for context_root_idx in context_root_indices:
            context_root_idx += len(question_graph)
            for question_root_idx in question_root_indices:
                assert question_root_idx not in combined_graph[context_root_idx]
                assert context_root_idx not in combined_graph[question_root_idx]
                combined_graph[context_root_idx][question_root_idx] = 'cross-type'
                combined_graph[question_root_idx][context_root_idx] = 'cross-type'
        # Sanity check
        for i in range(len(combined_graph)):
            assert len(combined_graph[i]) >= 2
            assert combined_graph[i][i] == 'self'
        combined_node_paths, combined_edge_paths = get_paths_on_graph(combined_graph)

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    one_more_sep = tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
    sequence_added_tokens = (tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1 if one_more_sep
                             else tokenizer.model_max_length - tokenizer.max_len_single_sentence)
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    use_cls = (config['model_type'] != 'mt5')
    cls_token_id = tokenizer.cls_token_id if use_cls else -1
    sep_token_id = tokenizer.sep_token_id if config['model_type'] != 'mt5' else tokenizer.eos_token_id

    # Encode entire doc into multiple segments with stride
    spans = []
    span_doc_tokens = context_subtokens  # Remaining context subtokens
    overlapping_in_overflowing = max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens
    while len(spans) * doc_stride < len(context_subtokens):
        # Use the padding side for truncation
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        # Encode sentence-pair
        encoded_dict = tokenizer.encode_plus(
            texts,
            pairs,
            truncation=truncation,
            padding='max_length',
            # padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,  # On the truncated side
            stride=overlapping_in_overflowing,
            return_token_type_ids=True  # For XLM-R, there is only one token type
        )

        # Get encoded context length of current sequence
        paragraph_start = len(spans) * doc_stride  # Start idx of context subtokens of current sequence
        paragraph_len = min(
            len(context_subtokens) - paragraph_start,  # Actual remaining length
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,  # Max allowable length
        )
        paragraph_end = paragraph_start + paragraph_len  # Exclusive

        # Get text of encoded sentence-pair without padding
        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]
        else:
            non_padded_ids = encoded_dict["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}  # Sequence idx to orig idx
        for i in range(paragraph_len):
            index = (len(truncated_query) + sequence_added_tokens + i) if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = context_tok_to_orig_idx[paragraph_start + i]

        def adapt_graph_paths_for_sequence():
            # [CLS] + truncated query + [SEP] (+ [SEP]) + encoded context + [SEP] + [PAD]...
            # Haven't added support for mT5
            if combined_graph is None:
                return None, None
            seq_node_paths, seq_edge_paths = [], []
            question_offset, context_offset = 1, 1 + len(truncated_query) + (2 if one_more_sep else 1)

            def add_edges_for_special(idx):
                node_paths = [(i,) for i in range(max_seq_length)]
                edge_paths = [('special',)] * max_seq_length
                edge_paths[idx] = ('self',)
                seq_node_paths.append(tuple(node_paths))
                seq_edge_paths.append(tuple(edge_paths))

            def adapt_path(node_path, edge_path):
                # offset: offset of question or context in current sequence
                new_node_path, new_edge_path = [], []
                for tok, rel in zip(node_path, edge_path):
                    if tok < len(truncated_query):
                        new_node_path.append(tok + question_offset)
                        new_edge_path.append(rel)
                    else:
                        tok -= len(truncated_query)  # Corresponding idx in context_subtokens/context_graphs
                        if paragraph_start <= tok < paragraph_end:
                            new_node_path.append(tok - paragraph_start + context_offset)
                            new_edge_path.append(rel)
                return tuple(new_node_path), tuple(new_edge_path)

            def build_edges_from_i(from_i):
                # from_i: corresponding idx in combined_graph
                node_paths, edge_paths = [], []
                # to CLS
                node_paths.append((0,))
                edge_paths.append(('special',))
                # to query
                for to_i in range(len(truncated_query)):
                    node_path, edge_path = combined_node_paths[from_i][to_i], combined_edge_paths[from_i][to_i]
                    node_path, edge_path = adapt_path(node_path, edge_path)
                    node_paths.append(node_path)
                    edge_paths.append(edge_path)
                # to SEP
                assert len(node_paths) == (1 + len(truncated_query))
                node_paths.append((len(node_paths),))
                edge_paths.append(('special',))
                if one_more_sep:
                    node_paths.append((len(node_paths),))
                    edge_paths.append(('special',))
                # to context in current sequence
                for to_i in range(paragraph_start, paragraph_end):
                    node_path = combined_node_paths[from_i][to_i + len(truncated_query)]
                    edge_path = combined_edge_paths[from_i][to_i + len(truncated_query)]
                    node_path, edge_path = adapt_path(node_path, edge_path)
                    node_paths.append(node_path)
                    edge_paths.append(edge_path)
                # to SEP
                assert len(node_paths) == (len(truncated_query) + paragraph_len + (3 if one_more_sep else 2))
                node_paths.append((len(node_paths),))
                edge_paths.append(('special',))
                # to PAD
                while len(node_paths) < max_seq_length:
                    node_paths.append((len(node_paths),))
                    edge_paths.append(('special',))
                return tuple(node_paths), tuple(edge_paths)

            # CLS
            add_edges_for_special(idx=0)
            # query
            for from_i in range(len(truncated_query)):
                node_paths, edge_paths = build_edges_from_i(from_i)
                seq_node_paths.append(node_paths)
                seq_edge_paths.append(edge_paths)
            # SEP
            assert len(seq_node_paths) == (1 + len(truncated_query))
            add_edges_for_special(idx=len(seq_node_paths))
            if one_more_sep:
                add_edges_for_special(idx=len(seq_node_paths))
            # context in current sequence
            for from_i in range(paragraph_start, paragraph_end):
                node_paths, edge_paths = build_edges_from_i(from_i + len(truncated_query))
                seq_node_paths.append(node_paths)
                seq_edge_paths.append(edge_paths)
            # SEP
            assert len(seq_node_paths) == (len(truncated_query) + paragraph_len + (3 if one_more_sep else 2))
            add_edges_for_special(idx=len(seq_node_paths))
            # PAD
            while len(seq_node_paths) < max_seq_length:
                add_edges_for_special(idx=len(seq_node_paths))

            return tuple(seq_node_paths), tuple(seq_edge_paths)

        def adapt_root_paths_for_sequence(use_cls):
            # [CLS] + truncated query + [SEP] (+ [SEP]) + encoded context + [SEP] + [PAD]...
            seq_node_paths, seq_edge_paths = [], []
            # CLS
            if use_cls:
                seq_node_paths.append((0,))
                seq_edge_paths.append(())
            use_cls = int(use_cls)
            # query
            for node_path, edge_path in zip(question_node_paths, question_edge_paths):
                node_path = tuple([i + use_cls for i in node_path])  # Adjust question tok offset
                seq_node_paths.append(node_path)
                seq_edge_paths.append(edge_path)
            # SEP
            assert len(seq_node_paths) == len(truncated_query) + use_cls
            seq_node_paths.append((len(seq_node_paths),))
            seq_edge_paths.append(())
            if one_more_sep:
                seq_node_paths.append((len(seq_node_paths),))
                seq_edge_paths.append(())
            # context in current sequence
            offset = len(seq_node_paths)
            for node_path, edge_path in zip(context_node_paths[paragraph_start: paragraph_end],
                                            context_edge_paths[paragraph_start: paragraph_end]):
                new_node_path, new_edge_path = [node_path[0] - paragraph_start + offset], []
                for i, rel in zip(node_path[1:], edge_path):
                    if paragraph_start <= i < paragraph_end:
                        new_node_path.append(i - paragraph_start + offset)  # Adjust context tok offset
                        new_edge_path.append(rel)
                seq_node_paths.append(tuple(new_node_path))
                seq_edge_paths.append(tuple(new_edge_path))
            # SEP
            assert len(seq_node_paths) == len(truncated_query) + paragraph_len + (2 if one_more_sep else 1) + use_cls
            seq_node_paths.append((len(seq_node_paths),))
            seq_edge_paths.append(())
            # PAD
            while len(seq_node_paths) < max_seq_length:
                seq_node_paths.append((len(seq_node_paths),))
                seq_edge_paths.append(())
            return tuple(seq_node_paths), tuple(seq_edge_paths)  # [seq len, path len]

        def adapt_graph_rel_for_sequence(use_cls):
            # [CLS] + truncated query + [SEP] (+ [SEP]) + encoded context + [SEP] + [PAD]...
            seq_graph_rels = []
            # CLS
            if use_cls:
                seq_graph_rels.append({0: 'self'})
            use_cls = int(use_cls)
            # query
            for edges in question_graph:
                seq_graph_rels.append({i + use_cls: rel for i, rel in edges.items()})  # Adjust question tok offset
            # SEP
            assert len(seq_graph_rels) == len(truncated_query) + use_cls
            seq_graph_rels.append({len(seq_graph_rels): 'self'})
            if one_more_sep:
                seq_graph_rels.append({len(seq_graph_rels): 'self'})
            # context in current sequence
            offset = len(seq_graph_rels)
            for edges in context_graph[paragraph_start: paragraph_end]:
                seq_graph_rels.append({i - paragraph_start + offset: rel for i, rel in edges.items()
                                       if paragraph_start <= i < paragraph_end})  # Adjust context tok offset
            # SEP
            assert len(seq_graph_rels) == len(truncated_query) + paragraph_len + (2 if one_more_sep else 1) + use_cls
            seq_graph_rels.append({len(seq_graph_rels): 'self'})
            # PAD
            while len(seq_graph_rels) < max_seq_length:
                seq_graph_rels.append({len(seq_graph_rels): 'self'})

            # Connect question and context roots
            for c_i in context_root_indices:
                if paragraph_start <= c_i < paragraph_end:
                    seq_c_i = c_i - paragraph_start + offset
                    for q_i in question_root_indices:
                        seq_q_i = q_i + use_cls
                        assert seq_q_i not in seq_graph_rels[seq_c_i], \
                            f'question root {seq_q_i} should not have been added for {seq_c_i}'
                        assert seq_c_i not in seq_graph_rels[seq_q_i], \
                            f'context root {seq_c_i} should not have been added for {seq_q_i}'
                        seq_graph_rels[seq_c_i][seq_q_i] = 'cross-type'
                        seq_graph_rels[seq_q_i][seq_c_i] = 'cross-type'

            return seq_graph_rels

        def adapt_features_for_sequence(question_feats, context_feats, default):
            # [CLS] + truncated query + [SEP] (+ [SEP]) + encoded context + [SEP] + [PAD]...
            seq_feats = []
            # CLS
            if use_cls:
                seq_feats.append(default)
            # query
            seq_feats += question_feats
            # SEP
            seq_feats.append(default)
            if one_more_sep:
                seq_feats.append(default)
            # context in current sequence
            seq_feats += context_feats[paragraph_start: paragraph_end]
            # SEP
            seq_feats.append(default)
            # PAD
            seq_feats += [default] * (max_seq_length - len(seq_feats))
            return tuple(seq_feats)

        # Build non-graph features for current sequence
        seq_pos = adapt_features_for_sequence(question_pos_ids, context_pos_ids, default=0)
        seq_depth = adapt_features_for_sequence(question_depth, context_depth, default=0)
        seq_rels = adapt_features_for_sequence(question_rels, context_rels, default='none')

        # Build graph paths for current sequence
        assert tokenizer.padding_side == 'right', 'Feature processing and postprocessing assume right side padding'
        seq_node_paths, seq_edge_paths = [], []
        if config['use_graph_path']:
            seq_node_paths, seq_edge_paths = adapt_graph_paths_for_sequence()
        elif config['use_root_path'] or True:
            seq_node_paths, seq_edge_paths = adapt_root_paths_for_sequence(use_cls)

        # Build graph rels for current sequence
        seq_graph_rels = []
        if config['use_graph_rel'] or True:
            seq_graph_rels = adapt_graph_rel_for_sequence(use_cls)

        # Sanity check on features: [CLS] + truncated query + [SEP] (+ [SEP]) + encoded context + [SEP] + [PAD]...
        input_ids = encoded_dict["input_ids"]
        # CLS
        if use_cls:
            assert input_ids[0] == cls_token_id
        use_cls = int(use_cls)
        # query
        assert input_ids[use_cls] == truncated_query[0]
        query_end_idx = len(truncated_query) - 1 + use_cls
        assert input_ids[query_end_idx] == truncated_query[-1]
        assert seq_pos[query_end_idx] == question_pos_ids[-1]
        assert seq_depth[query_end_idx] == question_depth[-1]
        assert seq_rels[query_end_idx] == question_rels[-1]
        # SEP
        assert input_ids[query_end_idx + 1] == sep_token_id
        # context in current sequence
        context_end = query_end_idx + paragraph_len + (2 if one_more_sep else 1) # Inclusive
        assert input_ids[context_end] == tokenizer.convert_tokens_to_ids(context_subtokens[paragraph_end - 1])
        assert seq_pos[context_end] == context_pos_ids[paragraph_end - 1]
        assert seq_depth[context_end] == context_depth[paragraph_end - 1]
        assert seq_rels[context_end] == context_rels[paragraph_end - 1]
        # SEP
        assert input_ids[context_end + 1] == sep_token_id

        # For use_graph_path; sanity check on graph paths
        if config['use_graph_path']:
            assert len(seq_node_paths) == len(seq_edge_paths)
            for from_i in range(max_seq_length):
                for to_i in range(max_seq_length):
                    assert len(seq_node_paths[from_i][to_i]) > 0, 'Must there be a path (can be partial)'
                    if from_i == to_i:
                        assert seq_node_paths[from_i][to_i] == (to_i,), 'Must there be a self path'
                        assert seq_edge_paths[from_i][to_i] == ('self',)
                    else:
                        assert seq_node_paths[from_i][to_i][-1] == to_i, 'Path should always ends with target'
                        assert len(seq_node_paths[from_i][to_i]) == len(seq_edge_paths[from_i][to_i])

        # For use_root_path; sanity check on root paths
        if config['use_root_path'] or True:
            assert len(seq_node_paths) == len(seq_edge_paths)
            for i, (node_path, edge_path) in enumerate(zip(seq_node_paths, seq_edge_paths)):
                assert len(node_path) == len(edge_path) + 1
                assert node_path[0] == i, f'Root path always starts from self: {seq_node_paths} vs {i}'
                if i <= query_end_idx:
                    assert all([node <= query_end_idx for node in node_path]), 'Question root path should be within question'
                else:
                    assert all([query_end_idx < node < max_seq_length for node in node_path]), 'Context root path should be within context'

        # For use_graph_rel; sanity check on graph rels
        if config['use_graph_rel'] or True:
            for i in range(max_seq_length):
                assert seq_graph_rels[i].get(i, 'none') == 'self', f'{i} does not have self connection'

        # Visualize graph path if needed
        debug = False
        if debug and paragraph_len < 80:
            if config['use_root_path'] or True:
                save_path = f'{join(config["log_dir"], "visual", "root_path")}'
                visualize_root_path(tokenizer, input_ids, seq_depth, seq_node_paths, seq_edge_paths, save_path)
                logger.info(f'Saved root path visualization to {save_path}.pdf; exit')
            if config['use_graph_rel'] or True:
                save_path = f'{join(config["log_dir"], "visual", "graph_rel")}'
                visualize_graph_rel(tokenizer, input_ids, seq_depth, seq_graph_rels, save_path)
                logger.info(f'Saved graph rel visualization to {save_path}.pdf; exit')
            exit(0)

        # Convert edge rel to id in feature
        seq_rels = [rel_to_id[rel] for rel in seq_rels]

        # For use_graph_path; convert edge rel to id in graph path
        if config['use_graph_path']:
            new_seq_edge_paths = []
            for from_i in range(max_seq_length):
                edge_paths = [tuple([rel_to_id(rel) for rel in seq_edge_paths[from_i][to_i]]) for to_i in
                              range(max_seq_length)]
                new_seq_edge_paths.append(tuple(edge_paths))
            seq_edge_paths = tuple(new_seq_edge_paths)

        # For use_root_path; convert edge rel to id in root path
        if config['use_root_path'] or True:
            new_seq_edge_paths = []
            for edge_path in seq_edge_paths:
                new_seq_edge_paths.append(tuple([rel_to_id[rel] for rel in edge_path]))
            seq_edge_paths = tuple(new_seq_edge_paths)

        # For use_graph_rel; convert rel to id and use sparse representation
        seq_graph_indices, seq_graph_rel_ids = [], []
        if config['use_graph_rel'] or True:
            for edges in seq_graph_rels:
                edge_indices, edge_rel_ids = [], []
                for i, rel in edges.items():
                    edge_indices.append(i)
                    edge_rel_ids.append(rel_to_id[rel])
                seq_graph_indices.append(tuple(edge_indices))
                seq_graph_rel_ids.append(tuple(edge_rel_ids))

        encoded_dict["start"] = paragraph_start  # Start idx of context subtokens of current sequence
        encoded_dict["paragraph_len"] = paragraph_len  # Encoded context length of current sequence
        encoded_dict["tokens"] = tokens  # Text of the encoded sentence-pair without padding
        encoded_dict["token_to_orig_map"] = token_to_orig_map  # Sequence idx to orig idx
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["pos"] = seq_pos
        encoded_dict["node_depth"] = seq_depth
        encoded_dict["node_rel"] = seq_rels
        encoded_dict["seq_node_paths"] = seq_node_paths
        encoded_dict["seq_edge_paths"] = seq_edge_paths
        encoded_dict["seq_graph_indices"] = seq_graph_indices
        encoded_dict["seq_graph_rel_ids"] = seq_graph_rel_ids

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    # Fill token_is_max_context
    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id) if use_cls else -1

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        # Excluded tokens: padding, [SEP], query
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens:] = 0
        else:
            p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        # If the example always have answers, filter out spans without gold answers
        span_is_impossible = example.is_impossible
        start_position = cls_index
        end_position = cls_index
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["paragraph_len"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
                # tokens[start_position: end_position + 1] should be the answer text

        features.append(
            SquadFeatures(
                span["input_ids"],  # Full encoded sentence-pair
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],  # Encoded context length of current sequence
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],  # Encoded sentence-pair without padding
                token_to_orig_map=span["token_to_orig_map"],  # Sequence idx to orig idx
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
                pos=tuple(span["pos"]),
                node_depth=tuple(span["node_depth"]),
                node_rel=tuple(span["node_rel"]),
                seq_node_paths=tuple(span["seq_node_paths"]),
                seq_edge_paths=tuple(span["seq_edge_paths"]),
                seq_graph_indices=tuple(span["seq_graph_indices"]),
                seq_graph_rel_ids=tuple(span["seq_graph_rel_ids"])
            )
        )
    return features


def squad_convert_example_to_features_init(tokenizer_for_convert, config_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

    global config
    config = config_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    config,
    padding_strategy="max_length",
    return_dataset=False,
    threads=4,
    tqdm_enabled=True,
):
    # Convert each example to multiple input features
    features = []
    threads = min(threads, cpu_count())
    not_use_thread = False  # For better debug without threads
    if not_use_thread:
        squad_convert_example_to_features_init(tokenizer, config)
        features = list(
            tqdm(
                [squad_convert_example_to_features(
                    example,
                    max_seq_length=max_seq_length,
                    doc_stride=doc_stride,
                    max_query_length=max_query_length,
                    padding_strategy=padding_strategy,
                    is_training=is_training
                ) for example in examples],
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )
    else:
        with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer, config)) as p:
            annotate_ = partial(
                squad_convert_example_to_features,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                padding_strategy=padding_strategy,
                is_training=is_training
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                    desc="convert squad examples to features",
                    disable=not tqdm_enabled,
                )
            )

    # Add example idx and unique id
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # Return dataset
    if return_dataset:
        dataset = QaDataset(config, features, is_training=is_training)
        return features, dataset
    else:
        return features


class SquadProcessor:
    """ Process squad-like dataset """
    def get_train_examples(self, data_dir, filename):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            input_data = json.load(f)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_or_test_examples(self, data_dir, filename):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            input_data = json.load(f)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry.get("title", "none")
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                context_parsed = paragraph.get("context_parsed", None)
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    question_parsed = qa.get("question_parsed", None)
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        question_parsed=question_parsed,
                        context_text=context_text,
                        context_parsed=context_parsed,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
        question_parsed=None,
        context_parsed=None
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0  # Inclusive

        # Get tokenized question
        tokenized = self._get_tokenized(question_text, question_parsed)
        self.question_tokens = tokenized['tokens']
        self.question_pos_tags = tokenized['pos_tags']
        self.question_heads = tokenized['heads']
        self.question_rels = tokenized['rels']
        self.question_sent_ids = tokenized['sent_ids']

        # Get tokenized context
        tokenized = self._get_tokenized(context_text, context_parsed)
        self.doc_tokens = tokenized['tokens']
        self.word_to_char_idx = tokenized['word_to_char_idx']
        self.pos_tags = tokenized['pos_tags']
        self.heads = tokenized['heads']
        self.rels = tokenized['rels']
        self.sent_ids = tokenized['sent_ids']  # Sentence splitting is not reliable for certain languages

        # Start and end positions only has a value during evaluation.
        char_to_word_offset = tokenized['char_to_word_offset']
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

    def _get_tokenized(self, text, text_parsed=None):
        # Tokenized by whitespace
        if not text_parsed:
            tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in text:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        tokens.append(c)
                    else:
                        tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(tokens) - 1)
            return {
                'tokens': tuple(tokens),
                'char_to_word_offset': tuple(char_to_word_offset),
                'word_to_char_idx': None,
                'pos_tags': None,
                'heads': None,
                'rels': None,
                'sent_ids': None
            }

        # Use tokenized from parsing; here we refer to tokens and words interchangeably
        if 'token_to_char_idx' not in text_parsed[0]:
            # Use char matching to get alignment between tokens and text
            # Only work for languages without MWT
            token_offset = 0
            tokens, pos_tags, heads, rels, sent_ids = [], [], [], [], []
            for sent_id, sent_parsed in enumerate(text_parsed):
                tokens += sent_parsed['tokens']
                pos_tags += sent_parsed['pos_tags']
                heads += [(h + token_offset) if h >= 0 else -1 for h in sent_parsed['heads']]  # Root: -1
                rels += sent_parsed['rels']
                sent_ids += [sent_id] * len(sent_parsed['tokens'])
                token_offset += len(sent_parsed['tokens'])

            char_to_word_offset, word_idx, word_char_idx = [], 0, 0
            word_to_char_idx, left_char_idx, prev_is_whitespace = [], 0, True  # left and right char idx: inclusive
            for c_idx, c in enumerate(text):
                if not _is_whitespace(c):
                    assert c == tokens[word_idx][
                        word_char_idx], f'{repr(c)} not in {repr(tokens[word_idx])}'  # idx should be valid
                    char_to_word_offset.append(word_idx)
                    word_char_idx += 1
                    if prev_is_whitespace:
                        left_char_idx = c_idx  # For case: ab cd -> [ab], [cd]
                    if word_char_idx >= len(tokens[word_idx]):
                        word_idx += 1  # When word_idx reach len(tokens), c should reach ends
                        word_char_idx = 0
                        word_to_char_idx.append((left_char_idx, c_idx))
                        assert tokens[word_idx - 1] == text[left_char_idx: c_idx + 1]
                        left_char_idx = c_idx + 1  # For case: abcd -> [ab], [cd]
                    prev_is_whitespace = False
                else:
                    char_to_word_offset.append(word_idx)  # Ending whitespace has word offset == len(tokens)
                    prev_is_whitespace = True
            assert word_idx >= len(tokens) - 1  # When text ends, word_idx should reach end
            assert len(word_to_char_idx) == len(tokens)

            return {
                'tokens': tuple(tokens),
                'char_to_word_offset': tuple([i if i < len(tokens) else len(tokens) - 1 for i in char_to_word_offset]),
                'word_to_char_idx': tuple(word_to_char_idx),
                'pos_tags': pos_tags,
                'heads': tuple(heads),
                'rels': tuple(rels),
                'sent_ids': tuple(sent_ids)
            }
        else:
            # Use the provided token to char mapping from preprocessing to align
            # Preprocessing takes care of MWT in the mapping
            token_offset = 0
            tokens, pos_tags, heads, rels, sent_ids, token_to_char_idx = [], [], [], [], [], []
            for sent_id, sent_parsed in enumerate(text_parsed):
                tokens += sent_parsed['tokens']
                pos_tags += sent_parsed['pos_tags']
                heads += [(h + token_offset) if h >= 0 else -1 for h in sent_parsed['heads']]  # Root: -1
                rels += sent_parsed['rels']
                sent_ids += [sent_id] * len(sent_parsed['tokens'])
                token_to_char_idx += [(left_char, right_char) for left_char, right_char in
                                      sent_parsed['token_to_char_idx']]
                token_offset += len(sent_parsed['tokens'])

            char_to_word_offset, char_match_count, word_idx = [], 0, 0
            for c_idx, c in enumerate(text):
                if word_idx == len(tokens):
                    char_to_word_offset.append(len(tokens) - 1)  # Ending whitespace has the last word offset
                elif c_idx < token_to_char_idx[word_idx][1]:
                    char_to_word_offset.append(word_idx)
                elif c_idx == token_to_char_idx[word_idx][1]:
                    char_to_word_offset.append(word_idx)
                    if c == tokens[word_idx][-1]:
                        char_match_count += 1
                    word_idx += 1
                    # Some MWT words that cannot be nicely aligned have same char idx; always select the first to map
                    while word_idx < len(tokens) and c_idx == token_to_char_idx[word_idx][1]:
                        word_idx += 1
                else:
                    raise ValueError(f'token_to_char_idx is not sequential {token_to_char_idx}')
            assert word_idx == len(tokens)  # text should exhaust tokens
            assert char_match_count >= len(tokens) * 0.5  # Most of chars should match word boundary (exception is MWT)

            return {
                'tokens': tuple(tokens),
                'char_to_word_offset': tuple(char_to_word_offset),
                'word_to_char_idx': tuple(token_to_char_idx),
                'pos_tags': pos_tags,
                'heads': tuple(heads),
                'rels': tuple(rels),
                'sent_ids': tuple(sent_ids)
            }


class SquadFeatures:
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        qas_id: str = None,
        pos=None,
        node_depth=None,
        node_rel=None,
        seq_node_paths=None,
        seq_edge_paths=None,
        seq_graph_indices=None,
        seq_graph_rel_ids=None
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id

        self.pos = pos
        self.node_depth = node_depth
        self.node_rel = node_rel
        self.seq_node_paths = seq_node_paths
        self.seq_edge_paths = seq_edge_paths
        self.seq_graph_indices = seq_graph_indices
        self.seq_graph_rel_ids = seq_graph_rel_ids


class SquadResult:
    def __init__(self, unique_id, start_logits, end_logits, tagging_logits=None,
                 start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.tagging_logits = tagging_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits
