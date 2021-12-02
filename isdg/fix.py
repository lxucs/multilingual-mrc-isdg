import json


# For vietnamese (vi), install a dev version; see # https://github.com/stanfordnlp/stanza/pull/535

def fix_mlqa_en_test(filepath):
    """ Fix space to avoid stanza error """
    with open(filepath, 'r') as f:
        data = json.load(f)

    for entry in data['data']:
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            if context.strip().startswith('In her later years, the Queen Mother became known for her longevity.'):
                print(f'Found entry')
                idx = context.index('On 5 March 2002, Queen Elizabeth')
                if idx == -1:
                    print('Error: cannot find substring')
                    return
                fixed_context = context[:idx] + ' ' + context[idx:]  # Insert space after sentence end
                assert len(paragraph['context']) + 1 == len(fixed_context)
                paragraph['context'] = fixed_context
                print('Fixed!')

    with open(filepath, 'w') as f:
        json.dump(data, f)


fix_mlqa_en_test('/home/lxu85/clsp_data_dir/download/mlqa/MLQA_V1/test/test-context-en-question-en.json')
