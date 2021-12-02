from os import makedirs
from os.path import join
import numpy as np
import pyhocon
import logging
import torch
import random
from transformers import BertTokenizer, XLMRobertaTokenizer, T5Tokenizer

logger = logging.getLogger(__name__)

langs = {
    'xquad': ['en', 'de', 'el', 'es', 'hi', 'ru'],
    'mlqa': ['en', 'de', 'es', 'hi'],
    'tydiqa': ['en', 'fi', 'ko', 'ru']
}

upos_to_id = {tag: i for i, tag in enumerate(
    ['X', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
     'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB'])}  # 0: other

deprels = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp',
           'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse', 'dislocated', 'expl',
           'fixed', 'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan',
           'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']  # Child to parent
special_rels = ['self', 'subword', 'sibling', 'ancestor', 'descendant', 'cross-sent', 'cross-type', 'special']

all_rels = ['none'] + deprels + [f'R-{rel}' for rel in deprels] + special_rels
rel_to_id = {rel: i for i, rel in enumerate(all_rels)}  # 0: no relation

unicode_whitespace = {'\u0020', '\u00A0', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004',
                      '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200A', '\u3000'}


def flatten(l):
    return [item for sublist in l for item in sublist]


def initialize_config(config_name, create_dir=True):
    logger.info("Experiment: {}".format(config_name))

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[config_name]

    config['log_dir'] = join(config["log_root"], config_name)
    config['tb_dir'] = join(config['log_root'], 'tensorboard')
    if create_dir:
        makedirs(config['log_dir'], exist_ok=True)
        makedirs(config['tb_dir'], exist_ok=True)

    assert not config['use_graph_path'], 'Do not support graph path because of memory size'
    assert not config['use_undirected_path'] or not config['use_expanded_value'], 'Cannot expand value if undirected'

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    logger.info('Random seed is set to %d' % seed)


def get_bert_tokenizer(config):
    # Avoid using fast tokenization
    if config['model_type'] == 'mt5':
        return T5Tokenizer.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'bert':
        return BertTokenizer.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlm-roberta':
        return XLMRobertaTokenizer.from_pretrained(config['pretrained'])
    else:
        raise ValueError('Unknown model type')


def random_select(tensor, num_selection):
    """ Randomly select first dimension """
    if tensor.shape[0] > num_selection:
        rand_idx = torch.randperm(tensor.shape[0])[:num_selection]
        return tensor[rand_idx]
    else:
        return tensor


def batch_select(tensor, idx):
    """ Do selection per row (first axis). """
    assert tensor.shape[0] == idx.shape[0]
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]
    view_shape = (dim0_size * dim1_size,) + tensor.shape[2:]
    tensor = tensor.view(*view_shape)

    idx_offset = torch.arange(0, dim0_size, device=tensor.device) * dim1_size
    idx_offset = idx_offset.unsqueeze(-1)
    new_idx = idx + idx_offset
    selected = tensor[new_idx]
    return selected
