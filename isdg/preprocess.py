from os.path import join
import stanza
import json
from tqdm import tqdm
import argparse
from util import langs


def parse_text(nlp, text, verbose=True):
    try:
        parsed = []
        doc = nlp(text)
        for sent in doc.sentences:  # Should be one sentence in most cases
            tokens = [tok.text for tok in sent.words]
            pos_tags = [tok.upos for tok in sent.words]
            heads = [tok.head - 1 for tok in sent.words]  # Root has head -1
            rels = [tok.deprel for tok in sent.words]
            sent_len = len(sent.text)

            # Build word to char idx mapping
            word_to_char_idx, aligned_word_count = [], 0
            for token in sent.tokens:
                if sum([len(word.text) for word in token.words]) == len(token.text):
                    # If words have total length equal to token length, align words as token split
                    left_char = token.start_char
                    for word_i, word in enumerate(token.words):
                        right_char = left_char + len(word.text) - 1  # Inclusive
                        if word_i == len(token.words) - 1 and right_char + 1 != token.end_char:
                            right_char = token.end_char - 1  # Correct right char if needed (shouldn't happen in theory)
                            if verbose:
                                print(f'Correct word right char to token right char: {word.text}, {token.text}')
                        word_to_char_idx.append((left_char, right_char))
                        if word.text == text[left_char: right_char + 1]:
                            aligned_word_count += 1
                        elif verbose:
                            print(f'Aligned word with changed text: {word.text} vs {text[left_char: right_char + 1]}')
                        left_char = right_char + 1
                else:
                    # If words cannot be aligned above, align all words the same as token char ends
                    word_to_char_idx += [(token.start_char, token.end_char - 1)] * len(token.words)
                    if verbose:
                        print(f'Words cannot be nicely aligned: {[word.text for word in token.words]} vs {token.text}')
            assert len(word_to_char_idx) == len(tokens)
            if verbose:
                print(f'Completely aligned words vs total tokens: {aligned_word_count} vs {len(sent.tokens)}')

            parsed.append({
                'tokens': tokens,
                'pos_tags': pos_tags,
                'heads': heads,
                'rels': rels,
                'sent_len': sent_len,
                'token_to_char_idx': word_to_char_idx
            })
        return parsed
    except Exception as e:
        print(f'Error on text: {text}')
        print(e)
        return None


def parse_file(filepath, lang, verbose):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stanza.download(lang)
    nlp = stanza.Pipeline(lang=lang)

    error_flag = False
    for entry in tqdm(data['data']):
        if error_flag:
            break
        for paragraph in entry['paragraphs']:
            if error_flag:
                break
            if True or 'context_parsed' not in paragraph:
                parsed = parse_text(nlp, paragraph['context'], verbose)
                if parsed:
                    paragraph['context_parsed'] = parsed
                else:
                    error_flag = True
                    break

            for qa in paragraph['qas']:
                if True or 'question_parsed' not in qa:
                    parsed = parse_text(nlp, qa['question'], verbose)
                    if parsed:
                        qa['question_parsed'] = parsed
                    else:
                        error_flag = True
                        break

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'Saved to {filepath}')


def parse_dataset(args):
    if args.dataset_name == 'squad':
        data_dir = join(args.download_dir, 'squad')
        for partition in ['train', 'dev']:
            parse_file(join(data_dir, f'{partition}-v1.1.json'), 'en', args.verbose)
    elif args.dataset_name == 'xquad':
        data_dir = join(args.download_dir, 'xquad')
        for lang in langs['xquad']:
            parse_file(join(data_dir, f'xquad.{lang}.json'), lang, args.verbose)
    elif args.dataset_name == 'mlqa':
        for partition in ['test', 'dev']:
            data_dir = join(args.download_dir, f'mlqa/MLQA_V1/{partition}')
            for lang in langs['mlqa']:
                parse_file(join(data_dir, f'{partition}-context-{lang}-question-{lang}.json'), lang, args.verbose)
    elif args.dataset_name == 'tydiqa':
        data_dir = join(args.download_dir, 'tydiqa', 'tydiqa-goldp-v1.1-train')
        file = 'tydiqa.en.train.json'
        parse_file(join(data_dir, file), 'en', args.verbose)
        data_dir = join(args.download_dir, 'tydiqa', 'tydiqa-goldp-v1.1-dev')
        for lang in langs['tydiqa']:
            parse_file(join(data_dir, f'tydiqa.{lang}.dev.json'), lang, args.verbose)
    else:
        raise ValueError(f'Unknown dataset {args.dataset_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True,
                        help='Download directory')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['squad', 'xquad', 'mlqa', 'tydiqa'])
    parser.add_argument('--verbose', action='store_true',
                        help='Turn on verbose logging')
    args = parser.parse_args()

    parse_dataset(args)
