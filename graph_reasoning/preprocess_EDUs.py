import json
import os.path
import re

splits = ['test', 'dev', 'train']
# , 'dev', 'train'
data_dirs = ['ControlDataset', 'logiqa', 'anli_v1.0/R1', 'anli_v1.0/R2', 'anli_v1.0/R3', ]


def reformate_save_input_file():
    prefixs = ['', 'mc160.', 'mc500.']
    suffixs = ['.txt', '_in_entail.txt', '_in_entail.txt']
    _data_dirs = ['Binary-FEVER-NLI', 'MC160-NLI', 'MC500-NLI']
    for data_dir, prefix, suffix in zip(_data_dirs, prefixs, suffixs):
        splits = ['test', 'dev', 'train']
        snts = []

        for split in splits:
            json_lines = []
            with open(os.path.join('dataset', data_dir, '{}{}{}'.format(prefix, split, suffix)), 'r',
                      encoding='utf-8') as f:
                lines = f.readlines()
            premise_cache = ''
            for id, line in enumerate(lines):
                label, premise, hypothesis = line.strip().split('\t')
                if not premise == premise_cache:
                    premise_snts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', premise)
                    for snt in premise_snts:
                        snts.append(snt)
                    premise_cache = premise
                snts.append(hypothesis)
                json_lines.append({
                    'id': '{}'.format(id),
                    'premise': premise,
                    'hypothesis': hypothesis,
                    'label': label.replace('not_entailment', 'non_entailment')
                })
            if not os.path.exists(os.path.join('data', data_dir)):
                os.mkdir(os.path.join('data', data_dir))
            with open(os.path.join('data', data_dir, '{}.jsonl'.format(split)), 'w', encoding='utf-8') as f:
                for jline in json_lines:
                    f.write(json.dumps(jline) + '\n')

        cut = len(snts) // 20 + 1
        if not os.path.exists('E:\java\DiscourseSimplification\data\{}'.format(data_dir)):
            os.mkdir('E:\java\DiscourseSimplification\data\{}'.format(data_dir))
            for dir_name in ['input', 'flat', 'output', 'default']:
                os.mkdir('E:\java\DiscourseSimplification\data\{}\{}'.format(data_dir, dir_name))
        for i in range(20):
            with open('E:\java\DiscourseSimplification\data\{}\input\{}_{}.txt'.format(data_dir,
                                                                                       data_dir.replace('/', '_'),
                                                                                       i), 'w',
                      encoding='utf-8') as f:
                for j in range(cut * i, min(cut * (i + 1), len(snts))):
                    f.write(snts[j] + '\n')


#  'ControlDataset','logiqa''anli_v1.0/R1', 'anli_v1.0/R2', 'anli_v1.0/R3',
def save_input_file():
    # with open('E:\java\Graphene\DiscourseSimplification\input.txt', 'w', encoding='utf-8') as f:
    #     print('clear input file!')
    data_dirs = ['logiQA2.0-NLI', 'control']
    for data_dir in data_dirs:
        snts = []

        suffix = 'jsonl' if 'anli' in data_dir else 'jsonl.txt'
        premise_key = 'context' if 'anli' in data_dir else 'premise'
        splits = ['dev', 'train'] if data_dir == 'logiqa' else ['test', 'dev', 'train']
        for split in splits:
            json_lines = []
            with open(os.path.join('dataset', data_dir, '{}.{}'.format(split, suffix)), 'r', encoding='utf-8') as f:
                lines = f.readlines()

            premise_cache = ''
            for line in lines:
                line_data = json.loads(line.strip())
                if not line_data[premise_key] == premise_cache:

                    premise_snts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', line_data[premise_key])
                    for snt in premise_snts:
                        snts.append(snt)
                    premise_cache = line_data[premise_key]
                snts.append(line_data['hypothesis'])
                json_lines.append({
                    'id': line_data['id'] if 'logi' in data_dir else line_data['uid'],
                    'premise': line_data[premise_key],
                    'hypothesis': line_data['hypothesis'],
                    'label': line_data['label'].replace('not-entailment', 'non_entailment')
                })

            if not os.path.exists(os.path.join('data', data_dir)):
                os.mkdir(os.path.join('data', data_dir))
            with open(os.path.join('data', data_dir, '{}.jsonl'.format(split)), 'w', encoding='utf-8') as f:
                for jline in json_lines:
                    f.write(json.dumps(jline) + '\n')

        cut = len(snts) // 20 + 1
        if not os.path.exists('E:\java\DiscourseSimplification\data\{}'.format(data_dir)):
            os.mkdir('E:\java\DiscourseSimplification\data\{}'.format(data_dir))
            for dir_name in ['input', 'flat', 'output', 'default']:
                os.mkdir('E:\java\DiscourseSimplification\data\{}\{}'.format(data_dir, dir_name))
        for i in range(20):
            with open('E:\java\DiscourseSimplification\data\{}\input\{}_{}.txt'.format(data_dir,
                                                                                       data_dir.replace('/', '_'),
                                                                                       i), 'w',
                      encoding='utf-8') as f:
                for j in range(cut * i, min(cut * (i + 1), len(snts))):
                    f.write(snts[j] + '\n')


def read_flat_file():
    for data_dir in data_dirs:
        for i in range(20):
            os.replace(
                'E:\java\DiscourseSimplification\data\{}\\flat\{}_{}.flat'.format(data_dir, data_dir.replace('/', '_'),
                                                                                  i),
                'data\\{}\\flat\\{}_{}.flat'.format(data_dir, data_dir.replace('\\', '_'), i))


def reformate_flat_file():
    for data_dir in data_dirs:
        subsentences_with_logic = {}
        for filename in os.listdir('E:\java\DiscourseSimplification\data\{}\\flat'.format(data_dir)):
            if not filename.endswith('.flat'):
                continue
            flat_datas = open(f'E:\java\DiscourseSimplification\data\{data_dir}\\flat/{filename}', 'r',
                              encoding='utf-8').read().split('\n')
            flat_datas = list(filter(lambda x: x.strip() != '', flat_datas))
            new_flat_datas = []
            for data in flat_datas:
                if data not in new_flat_datas:
                    new_flat_datas.append(data)
            flat_datas = new_flat_datas
            for line in flat_datas:
                line = line.split('\t')
                origin_sentence, hash_id, num, subsentence = line[:4]
                logic_relation = line[4:]
                logic_relation = list(map(lambda x: x.split('('), logic_relation))
                logic_relation = list(
                    map(lambda x: (x[0].replace('L:', '').replace('S:', ''), x[1].replace(')', '')), logic_relation))
                origin_sentence = origin_sentence.replace('baseDir', '')
                if ord('a') <= ord(origin_sentence[-1]) <= ord('z') or ord('A') <= ord(origin_sentence[-1]) <= ord('Z'):
                    origin_sentence += '.'
                origin_sentence = origin_sentence.replace('ttherefore', 'therefore').replace('Ttherefore', 'Therefore')
                origin_sentence = origin_sentence.replace('<b>', '').replace('baseDir', '').replace('</b>', '').replace(
                    '<i>', '').replace('</i>', '').replace(
                    '<a>', '').replace('</a>', '').replace('baseDir', '').strip()
                if origin_sentence not in subsentences_with_logic:
                    subsentences_with_logic[origin_sentence] = {}
                subsentences_with_logic[origin_sentence][hash_id] = {'subsentence': subsentence,
                                                                     'logic_relation': logic_relation}


def reformate_EDUs():
    max_key_len = 256
    data_dirs = ['control']
    for data_dir in data_dirs:
        data = {}
        for filename in os.listdir('E:\java\DiscourseSimplification\data\{}\\output'.format(data_dir)):
            if not filename.endswith('.json'):
                continue
            split_datas = json.loads(open(f'E:\java\DiscourseSimplification/data/{data_dir}\\output/{filename}', 'r',
                                          encoding='utf-8').read())
            for snt_data in split_datas['sentences']:
                key = snt_data['originalSentence'].replace('<b>', '').replace('baseDir', '').replace('</b>', ''). \
                    replace('<i>', '').replace('</i>', '').replace(
                    'Ttherefore', 'Therefore').replace('ttherefore', 'therefore').strip()
                key = key[:max_key_len]
                data[key] = snt_data
        with open("data/{}/{}_EDUs.json".format(data_dir, data_dir.replace('\\', '_')), 'w', encoding='utf-8') as f:
            json.dump(data, f)


if __name__ == '__main__':
    # reformate_save_input_file()
    save_input_file()
    # read_flat_file()
    # reformate_EDUs()
