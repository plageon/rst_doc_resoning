import copy
import json
import random
import re
import sys

import dgl
import networkx as nx
import numpy as np
import pylev

import Config
from AdaLoGN.utils import pos_neg_convert, get_edge_norm, move_node_sentences, merge

def get_from_new_not_sentence_map(s, new_not_sentence_map, nlp):
    if s in new_not_sentence_map:
        return new_not_sentence_map[s]
    ret = pos_neg_convert(s, nlp, return_sentiment=False, return_adv_neg=True)
    new_not_sentence_map[s] = ret
    return ret


def get_element_map(text, EDUs, split_snts=False):
    def _slice(s: str) -> str:
        if '\n' in s:
            s = s[:s.find('\n')]
        s = s.strip()[:256]
        if 'They might well be overoptimistic , however, since corporate executives have' in s:
            s = "They might well be overoptimistic , however, since corporate executives have sometimes bought shares " \
                "in their own company in a calculated attempt to dispel negative rumors about the company' s health. " \
                "They might well be overoptimistic , however, since corp "
        return s.strip()

    if split_snts:
        element_map = {}
        snts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        for snt in snts:
            assert _slice(snt) in EDUs, snt
            text_prefix = _slice(snt).strip()
            element_map = {**element_map, **EDUs[text_prefix]['elementMap']}
    # element_map = {}
    # for sentence in EDUs[text_prefix]['sentences']:
    #     element_map = {**element_map, **sentence['elementMap']}
    else:
        assert _slice(text) in EDUs, text
        text_prefix = _slice(text)
        element_map = EDUs[text_prefix]['elementMap']
    del_hash_ids = []
    for hash_id in element_map:
        if len(element_map[hash_id]['text'].split(' ')) < 3 and len(element_map) > 1:
            del_hash_ids.append(hash_id)
    for hash_id in del_hash_ids:
        del element_map[hash_id]
    element_map_str = json.dumps(element_map)
    for hash_id in del_hash_ids:
        element_map_str = element_map_str.replace(hash_id, 'deleted')
    return edu_merge(json.loads(element_map_str))


def is_dgl_graph_connected(graph, show_graph=False):
    ret = nx.is_connected(dgl.to_networkx(graph).to_undirected())
    return ret


def sentence_format(s):
    s = s.replace(' \' s', '\'s').replace(' .', '.')
    return s


def random_delete_edges(context, answers, graphs, _node_sentences_a, _node_sentences_b, relations,
                        min_edge_nums=Config.truncate_edges_num):
    assert len(_node_sentences_a) + len(_node_sentences_b) == graphs.num_nodes()
    loop_count = 0
    while graphs.num_edges() > min_edge_nums and loop_count < 60:
        loop_count += 1
        del_edge_id = random.randint(0, graphs.num_edges() - 1)
        edges_a, edges_b = graphs.edges()[0].numpy().tolist(), graphs.edges()[1].numpy().tolist()

        edges_a_tmp = edges_a[:del_edge_id] + edges_a[del_edge_id + 1:]
        edges_b_tmp = edges_b[:del_edge_id] + edges_b[del_edge_id + 1:]
        del_node_a, del_node_b = edges_a[del_edge_id], edges_b[del_edge_id]

        new_graph = dgl.graph((edges_a_tmp, edges_b_tmp))
        if not is_dgl_graph_connected(new_graph, show_graph=False) or not new_graph.num_nodes() == graphs.num_nodes():
            continue

        graphs = new_graph
        del relations[del_edge_id]
    assert len(_node_sentences_a) + len(_node_sentences_b) == graphs.num_nodes()

    return context, answers, graphs, _node_sentences_a, _node_sentences_b, relations, None


def edu_merge(edus):
    new_edus = copy.deepcopy(edus)
    processed_texts = []
    del_hash_ids_map = {}
    for hash_id in edus:
        text = edus[hash_id]['text']
        if text in processed_texts:
            continue
        linked_contexts = []
        del_hash_ids = []
        for hash_id2 in edus:
            if edus[hash_id2]['text'] == text:
                linked_contexts += edus[hash_id2]['linkedContexts']
                del new_edus[hash_id2]
                del_hash_ids.append(hash_id2)
        element_map = edus[hash_id]
        element_map['linkedContexts'] = linked_contexts
        new_edus[hash_id] = element_map
        processed_texts.append(text)
        del_hash_ids_map[hash_id] = del_hash_ids
    edus_str = json.dumps(new_edus)
    for hash_id in del_hash_ids_map:
        for old_hash_id in del_hash_ids_map[hash_id]:
            edus_str = edus_str.replace(old_hash_id, hash_id)
    return json.loads(edus_str)


def construct_reasoning_graph_new(premise, hypothesis, EDUs, new_not_sentence_map, nlp):
    premise = premise.replace('<b>', '').replace('baseDir', '').replace('</b>', '').replace('<i>', '').replace('</i>',
                                                                                                               '').replace(
        'Ttherefore', 'Therefore').replace('ttherefore', 'therefore').replace('\n.', '').strip()
    relation_types = ['AND', 'OR', 'IMP', 'IMP_REV', 'OTHER', 'NOT']
    context_edus = get_element_map(premise, EDUs, True)

    # context, answers, graphs, node_sentences_a, node_sentences_b, relations

    sentences_a, sentences_b = [], []
    main_chains_sentences = []
    raw_relations = {}

    def add_nodes(_text_hash_id, a_or_b):
        if _text_hash_id in sentences_a and a_or_b == 'a':
            return
        if _text_hash_id in sentences_b and a_or_b == 'b':
            return
        if a_or_b == 'a':
            sentences_a.append(_text_hash_id)
        elif a_or_b == 'b':
            sentences_b.append(_text_hash_id)
        else:
            raise NotImplementedError

    def add_edges(_text_hash_id_src, _text_hash_id_dst, relation_id):
        if f'{_text_hash_id_src}_{_text_hash_id_dst}' not in raw_relations:
            raw_relations[f'{_text_hash_id_src}_{_text_hash_id_dst}'] = relation_id

    answer_sentence = hypothesis.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>',
                                                                                                   '').replace(
        'Ttherefore', 'Therefore').replace('ttherefore', 'therefore').replace('baseDir', '').replace('\n.', '').strip()
    answer_edus = get_element_map(answer_sentence, EDUs)
    sentences_a = list(context_edus.keys())
    sentences_b = list(answer_edus.keys())
    edus = {**context_edus, **answer_edus}

    for hash_id in edus:
        main_chains_sentences.append(hash_id)

    for hash_id, edu_data in edus.items():
        for linked_edu in edu_data['linkedContexts']:
            target_id = linked_edu['targetID']
            relation = linked_edu['relation']

            if target_id == 'deleted':
                continue
            if relation in ['UNKNOWN_SUBORDINATION', 'ATTRIBUTION', 'SPATIAL']:
                continue
            elif relation in ['BACKGROUND', 'CAUSE', 'CONDITION', 'PURPOSE', 'CAUSE_C']:
                add_nodes(target_id,
                          'a' if target_id in context_edus else ('b' if target_id in answer_edus else 'error'))
                add_edges(hash_id, target_id, relation_types.index('IMP'))
                add_edges(target_id, hash_id, relation_types.index('IMP_REV'))
            elif relation in ['RESULT', 'RESULT_C']:
                add_nodes(target_id,
                          'a' if target_id in context_edus else ('b' if target_id in answer_edus else 'error'))
                add_edges(target_id, hash_id, relation_types.index('IMP'))
                add_edges(hash_id, target_id, relation_types.index('IMP_REV'))
            elif relation in ['LIST', 'CONTRAST']:
                add_nodes(target_id,
                          'a' if target_id in context_edus else ('b' if target_id in answer_edus else 'error'))
                add_edges(target_id, hash_id, relation_types.index('AND'))
                add_edges(hash_id, target_id, relation_types.index('AND'))
            elif relation in ['DISJUNCTION']:
                add_nodes(target_id,
                          'a' if target_id in context_edus else ('b' if target_id in answer_edus else 'error'))
                add_edges(target_id, hash_id, relation_types.index('OR'))
                add_edges(hash_id, target_id, relation_types.index('OR'))
            elif relation in ['IDENTIFYING_DEFINITION', 'ELABORATION', 'DESCRIBING_DEFINITION',
                              'TEMPORAL_BEFORE_C', 'TEMPORAL_AFTER_C', 'TEMPORAL_BEFORE',
                              'TEMPORAL_AFTER']:
                continue
            elif relation in ['NOUN_BASED']:
                continue
            else:
                raise NotImplementedError

    for main_chain_id in range(len(main_chains_sentences) - 1):
        add_edges(main_chains_sentences[main_chain_id], main_chains_sentences[main_chain_id + 1],
                  relation_types.index('OTHER'))
        add_edges(main_chains_sentences[main_chain_id + 1], main_chains_sentences[main_chain_id],
                  relation_types.index('OTHER'))
    all_sentences = sentences_a + sentences_b
    # edges = [(all_sentences.index(id.split('_')[0]), all_sentences.index(id.split('_')[1])) for id in raw_relations]
    edges, relations = [], []
    for id in raw_relations:
        snt_1, snt_2 = id.split('_')
        for i in range(len(all_sentences)):
            if not snt_1 == all_sentences[i]:
                continue
            for j in range(len(all_sentences)):
                if snt_2 == all_sentences[j]:
                    edges.append((i, j))
                    relations.append(raw_relations[id])
    edges = ([s[0] for s in edges], [s[1] for s in edges])
    sentences_a = [sentence_format(edus[hash_id]['text']) for hash_id in sentences_a]
    sentences_b = [sentence_format(edus[hash_id]['text']) for hash_id in sentences_b]
    all_sentences = sentences_a + sentences_b

    nodes_num = len(all_sentences)
    edges_a, edges_b = edges[0], edges[1]
    for index1 in range(nodes_num):
        sentence_index1 = all_sentences[index1]
        for index2 in range(index1 + 1, nodes_num, 1):
            sentence_index2 = all_sentences[index2]
            if pylev.levenschtein(sentence_index1.split(' '), sentence_index2.split(' ')) > 2:
                continue
            if sentence_index1 in get_from_new_not_sentence_map(
                    sentence_index2, new_not_sentence_map, nlp) or sentence_index2 in get_from_new_not_sentence_map(
                sentence_index1, new_not_sentence_map, nlp):
                if [index1, index2] in [[a, b] for a, b in zip(edges_a, edges_b)]:
                    continue
                edges_a.append(index1)
                edges_b.append(index2)
                edges_a.append(index2)
                edges_b.append(index1)
                relations.append(relation_types.index('NOT'))

    graph = dgl.graph((edges_a, edges_b))
    assert is_dgl_graph_connected(graph)
    assert len(sentences_a) + len(sentences_b) == graph.num_nodes()
    return premise, hypothesis, graph, sentences_a, sentences_b, relations


def construct_reasoning_graph_extension(uid: str, context, answers, new_not_sentence_map, graphs=None,
                                        node_sentences_a=None,
                                        node_sentences_b=None, relations=None,
                                        return_base_nodes=False, nlp=None):
    assert context is not None

    base_node_ids = list(range(len(node_sentences_a) + len(node_sentences_b)))

    context = context.replace('<b>', '').replace('baseDir', '').replace('</b>', '').replace('<i>', '').replace('</i>',
                                                                                                               '').replace(
        'baseDir', '').strip()
    answers = answers.replace('<b>', '').replace('baseDir', '').replace('</b>', '').replace('<i>', '').replace('</i>',
                                                                                                               '').strip()

    node_sentences_c = []
    node_sentences_d = []
    relation_types = ['AND', 'OR', 'IMP', 'IMP_REV', 'OTHER', 'NOT']

    cont_exten_node_ids = []
    trans_exten_edge_ids = []

    def add_edge(_a, _b, _edge_type):
        assert _edge_type in relation_types
        assert 0 <= _a <= len(node_sentences_a) + len(node_sentences_b) + len(node_sentences_c) + len(
            node_sentences_d)
        assert 0 <= _b <= len(node_sentences_a) + len(node_sentences_b) + len(node_sentences_c) + len(
            node_sentences_d)
        graphs.add_edge(_a, _b)
        relations.append(relation_types.index(_edge_type))

    def add_node(_node_sentence, sentence_type):
        if _node_sentence in node_sentences_a or _node_sentence in node_sentences_b or _node_sentence in \
                node_sentences_c or _node_sentence in node_sentences_d:
            return (node_sentences_a + node_sentences_b + node_sentences_c + node_sentences_d).index(
                _node_sentence)
        assert sentence_type != 'EXIST'
        if sentence_type == 'c':
            node_sentences_c.append(_node_sentence)
        elif sentence_type == 'd':
            node_sentences_d.append(_node_sentence)
        else:
            raise NotImplementedError
        return len(node_sentences_a) + len(node_sentences_b) + len(node_sentences_c) + len(
            node_sentences_d) - 1

    imp_relation_id = relation_types.index('IMP')
    or_relation_id = relation_types.index('OR')
    and_relation_id = relation_types.index('AND')
    other_relation_id = relation_types.index('OTHER')

    tmp_edges_id = []
    node_sentences_a = [s + '_c' for s in node_sentences_a]
    node_sentences_b = [s + '_d' for s in node_sentences_b]

    max_extension_depth = 2

    for relation_index in range(len(relations)):
        if len(trans_exten_edge_ids) > Config.extension_padding_len:
            break
        if relations[relation_index] == other_relation_id:
            edge_a, edge_b = graphs.edges()[0].numpy().tolist()[relation_index], \
                graphs.edges()[1].numpy().tolist()[relation_index]
            for edge_index in range(len(graphs.edges()[0].numpy().tolist())):
                if graphs.edges()[0].numpy().tolist()[edge_index] == edge_b and relations[edge_index] in [
                    imp_relation_id, and_relation_id, or_relation_id]:
                    trans_exten_edge_ids.append(
                        [edge_a, graphs.edges()[1].numpy().tolist()[edge_index], relations[edge_index]])

    for _ in range(max_extension_depth - 1):
        if len(trans_exten_edge_ids) > Config.extension_padding_len:
            break
        for relation_index in range(len(relations)):
            if relations[relation_index] == imp_relation_id:
                edge_a, edge_b = graphs.edges()[0].numpy().tolist()[relation_index], \
                    graphs.edges()[1].numpy().tolist()[relation_index]
                for edge_index in range(len(graphs.edges()[0].numpy().tolist())):
                    if graphs.edges()[0].numpy().tolist()[edge_index] == edge_b and (
                            relations[edge_index] == imp_relation_id):
                        trans_exten_edge_ids.append(
                            [edge_a, graphs.edges()[1].numpy().tolist()[edge_index], imp_relation_id])
        edges_a, edges_b = graphs.edges()[0].numpy().tolist(), graphs.edges()[1].numpy().tolist()
        for index in range(len(trans_exten_edge_ids)):
            tmp_edges_id.append(len(edges_a))
            edges_a.append(trans_exten_edge_ids[index][0])
            edges_b.append(trans_exten_edge_ids[index][1])
            relations.append(trans_exten_edge_ids[index][2])
        graphs = dgl.graph((edges_a, edges_b))
        assert is_dgl_graph_connected(graphs)

    for relation_index in range(len(relations)):
        if len(cont_exten_node_ids) + len(trans_exten_edge_ids) > Config.extension_padding_len:
            break
        if relations[relation_index] == imp_relation_id:
            edge_a, edge_b = graphs.edges()[0].numpy().tolist()[relation_index], \
                graphs.edges()[1].numpy().tolist()[relation_index]
            sentence_a = (node_sentences_a + node_sentences_b)[edge_a].replace('_c', '').replace('_d', '')
            sentence_b = (node_sentences_a + node_sentences_b)[edge_b].replace('_c', '').replace('_d', '')

            sentence_a_neg = get_from_new_not_sentence_map(sentence_a, new_not_sentence_map, nlp)
            sentence_b_neg = get_from_new_not_sentence_map(sentence_b, new_not_sentence_map, nlp)

            if len(sentence_a_neg) == 0 or len(sentence_b_neg) == 0 or sentence_a_neg[0] == 'None' or \
                    sentence_b_neg[
                        0] == 'None' or 'error when convert' in sentence_a_neg or 'error when convert' in sentence_b_neg:
                continue
            not_a_node_id = add_node(sentence_a_neg[0] + ('_c' if edge_a < len(node_sentences_a) else '_d'),
                                     'c' if edge_a < len(node_sentences_a) else 'd')
            not_b_node_id = add_node(sentence_b_neg[0] + ('_c' if edge_b < len(node_sentences_a) else '_d'),
                                     'c' if edge_b < len(node_sentences_a) else 'd')

            add_edge(edge_a, not_a_node_id, 'NOT')
            add_edge(not_a_node_id, edge_a, 'NOT')
            add_edge(edge_b, not_b_node_id, 'NOT')
            add_edge(not_b_node_id, edge_b, 'NOT')
            add_edge(not_b_node_id, not_a_node_id, 'IMP')
            add_edge(not_a_node_id, not_b_node_id, 'IMP_REV')
            cont_exten_node_ids.append([edge_a, not_a_node_id, edge_b, not_b_node_id])

    edges_a, edges_b = graphs.edges()[0].numpy().tolist(), graphs.edges()[1].numpy().tolist()
    for edge_id in tmp_edges_id:
        edges_a[edge_id] = edges_b[edge_id] = relations[edge_id] = -1
    edges = (list(filter(lambda x: x != -1, edges_a)), list(filter(lambda x: x != -1, edges_b)))
    relations = list(filter(lambda x: x != -1, relations))

    new_edges, new_node_sentences_a, new_node_sentences_b, _base_node_id, _cont_exten_node_id, _trans_exten_edge_id = move_node_sentences(
        edges, node_sentences_a, node_sentences_b, node_sentences_c, node_sentences_d, base_node_ids,
        cont_exten_node_ids, trans_exten_edge_ids)
    base_node_ids = _base_node_id
    cont_exten_node_ids = _cont_exten_node_id
    trans_exten_edge_ids = _trans_exten_edge_id
    node_sentences_a = [s.replace('_c', '').strip() for s in new_node_sentences_a]
    node_sentences_b = [s.replace('_d', '').strip() for s in new_node_sentences_b]
    graphs = dgl.graph(new_edges)

    if return_base_nodes:
        return context, answers, graphs, node_sentences_a, node_sentences_b, relations, get_edge_norm(relations, graphs,
                                                                                                      graphs.edges()), base_node_ids, cont_exten_node_ids, trans_exten_edge_ids

    return context, answers, graphs, node_sentences_a, node_sentences_b, relations, get_edge_norm(relations, graphs,
                                                                                                  graphs.edges())


def construct_relation_graph_merger_nodes(id, show_graph=False, graph_type=4, context=None, answers=None,
                                          graphs=None, _node_sentences_a=None, _node_sentences_b=None, relations=None,
                                          selected_i=None, merge_4=False, return_merge_nodes=True, forced_merge=False,
                                          base_node_ids=None):
    if context is None:
        assert 1 <= graph_type <= 5
        assert id is not None
        context, answers, graphs, _node_sentences_a, _node_sentences_b, relations, _ = getattr(sys.modules[__name__],
                                                                                               f'construct_relation_graph{graph_type}')(
            id=id, show_graph=show_graph)

    ret_merge_nodes = []
    node_sentences_a = _node_sentences_a
    node_sentences_b = _node_sentences_b

    merge_nodes, edges = merge(graphs, relations, merge_relation_type=4,
                               node_sentence_a_len=len(node_sentences_a), forced_merge=forced_merge,
                               base_node_ids=base_node_ids)

    ret_merge_nodes.append(merge_nodes)
    for node in merge_nodes:
        if node < len(node_sentences_a):
            node_sentences_a[node] = node_sentences_a[node] + ' ' + node_sentences_a[node + 1]
            node_sentences_a[node + 1] = 'NULL'
        else:
            _node = node - len(node_sentences_a)
            node_sentences_b[_node] = node_sentences_b[_node] + ' ' + node_sentences_b[_node + 1]
            node_sentences_b[_node + 1] = 'NULL'

    new_node_num = graphs.num_nodes() - len(merge_nodes)

    for index in range(new_node_num):
        while (index not in edges[0] + edges[1]) and len(edges[0] + edges[1]) > 0 and index <= max(
                edges[0] + edges[1]):
            for edge_index in range(len(edges[0])):
                if edges[0][edge_index] > index:
                    edges[0][edge_index] -= 1
                if edges[1][edge_index] > index:
                    edges[1][edge_index] -= 1

    node_sentences_a = list(filter(lambda x: x != 'NULL', node_sentences_a))
    node_sentences_b = list(filter(lambda x: x != 'NULL', node_sentences_b))

    for index1 in range(len(edges[0])):
        if edges[0][index1] == -1:
            continue
        for index2 in range(index1 + 1, len(edges[0])):
            if edges[0][index1] == edges[0][index2] and edges[1][index1] == edges[1][index2]:
                edges[0][index2] = edges[1][index2] = -1
                relations[index2] = -1

    edges = (list(filter(lambda x: x != -1, edges[0])), list(filter(lambda x: x != -1, edges[1])))
    graphs = dgl.graph(edges)
    relations = list(filter(lambda x: x != -1, relations))

    _node_sentences_a = node_sentences_a
    _node_sentences_b = node_sentences_b

    assert len(node_sentences_a) != 0
    assert len(node_sentences_b) != 0

    if return_merge_nodes:
        return context, answers, graphs, _node_sentences_a, _node_sentences_b, relations, [
            get_edge_norm(relations, graphs, graphs.edges())], ret_merge_nodes
    return context, answers, graphs, _node_sentences_a, _node_sentences_b, relations, [get_edge_norm(r, g, g.edges())
                                                                                       for r, g in
                                                                                       zip(relations, graphs)]


def tie_nodes_ids(base_node_ids, cont_exten_node_ids, merge_nodes, trans_exten_edge_ids):
    if base_node_ids is None:
        return base_node_ids, cont_exten_node_ids, trans_exten_edge_ids

    def larger_count(l: list, v: int):
        l = np.array(l)
        return l[l < v].size

    for node in merge_nodes:
        if node in base_node_ids:
            base_node_ids[base_node_ids.index(node + 1)] = -1
    base_node_ids = list(filter(lambda x: x != -1, base_node_ids))

    for index in range(len(base_node_ids)):
        base_node_ids[index] -= larger_count(merge_nodes, base_node_ids[index])
    for index in range(len(cont_exten_node_ids)):
        for inner_index in range(4):
            cont_exten_node_ids[index][inner_index] -= larger_count(merge_nodes,
                                                                    cont_exten_node_ids[index][inner_index])
    for index in range(len(trans_exten_edge_ids)):
        for inner_index in range(2):
            trans_exten_edge_ids[index][inner_index] -= larger_count(merge_nodes,
                                                                     trans_exten_edge_ids[index][inner_index])

    return base_node_ids, cont_exten_node_ids, trans_exten_edge_ids


def construct_reasoning_graph(id: str, premise, hypothesis, EDUs, new_not_sentence_map, return_base_nodes=False,
                              min_edge_nums=Config.truncate_edges_num, dataset_dir=None, nlp=None):
    context, answers, graphs, node_sentences_a, node_sentences_b, relations = \
        construct_reasoning_graph_new(premise, hypothesis, EDUs, new_not_sentence_map, nlp)
    assert len(node_sentences_a) + len(node_sentences_b) == graphs.num_nodes()
    assert is_dgl_graph_connected(graphs)
    context, answers, graphs, node_sentences_a, node_sentences_b, relations, _, base_node_ids, cont_exten_node_ids, trans_exten_edge_ids = construct_reasoning_graph_extension(
        uid=id, context=context, answers=answers, new_not_sentence_map=new_not_sentence_map, graphs=graphs,
        node_sentences_a=node_sentences_a,
        node_sentences_b=node_sentences_b,
        relations=relations, return_base_nodes=return_base_nodes, nlp=nlp)
    is_dgl_graph_connected(graphs)

    for j in range(len(trans_exten_edge_ids)):
        assert len(trans_exten_edge_ids[j]) == 3

    count = 0
    while (graphs.num_nodes() > Config.truncate_nodes_num) and count < 3:
        context, answers, graphs, node_sentences_a, node_sentences_b, relations, _, merge_nodes = construct_relation_graph_merger_nodes(
            id=None, show_graph=False, graph_type=-1, context=context, answers=answers, graphs=graphs,
            _node_sentences_a=node_sentences_a, _node_sentences_b=node_sentences_b, relations=relations,
            selected_i=None, return_merge_nodes=True, base_node_ids=base_node_ids)
        count += 1
        base_node_ids, cont_exten_node_ids, trans_exten_edge_ids = tie_nodes_ids(base_node_ids, cont_exten_node_ids,
                                                                                 merge_nodes, trans_exten_edge_ids)
    assert is_dgl_graph_connected(graphs)
    assert len(node_sentences_a) + len(node_sentences_b) == graphs.num_nodes()
    if 'LogiQA' in dataset_dir:
        count = 0
        while graphs.num_nodes() > Config.truncate_nodes_num and count < 20:
            context, answers, graphs, node_sentences_a, node_sentences_b, relations, _, merge_nodes = construct_relation_graph_merger_nodes(
                id=None, show_graph=False, graph_type=-1, context=context, answers=answers, graphs=graphs,
                _node_sentences_a=node_sentences_a, _node_sentences_b=node_sentences_b, relations=relations,
                selected_i=None, return_merge_nodes=True, forced_merge=True, base_node_ids=base_node_ids)
            count += 1
            base_node_ids, cont_exten_node_ids, trans_exten_edge_ids = tie_nodes_ids(base_node_ids,
                                                                                     cont_exten_node_ids,
                                                                                     merge_nodes,
                                                                                     trans_exten_edge_ids)
            assert len(node_sentences_a) + len(node_sentences_b) == graphs.num_nodes()
            if graphs.num_edges() > Config.truncate_edges_num:
                context, answers, graphs, node_sentences_a, node_sentences_b, relations, _ = random_delete_edges(
                    context=context, answers=answers, graphs=graphs, _node_sentences_a=node_sentences_a,
                    _node_sentences_b=node_sentences_b, relations=relations,
                    min_edge_nums=min_edge_nums)
                assert len(node_sentences_a) + len(node_sentences_b) == graphs.num_nodes()
    if graphs.num_edges() > min_edge_nums:
        context, answers, graphs, node_sentences_a, node_sentences_b, relations, _ = random_delete_edges(
            context=context, answers=answers, graphs=graphs, _node_sentences_a=node_sentences_a,
            _node_sentences_b=node_sentences_b, relations=relations, min_edge_nums=min_edge_nums)

    assert is_dgl_graph_connected(graphs), id
    assert len(node_sentences_a) + len(node_sentences_b) == graphs.num_nodes()

    assert is_dgl_graph_connected(graphs), id
    edges_a, edges_b = graphs.edges()[0].numpy().tolist(), graphs.edges()[1].numpy().tolist()
    for index in range(len(edges_a)):
        if edges_a[index] == edges_b[index]:
            # self loop
            edges_a[index] = edges_b[index] = -1
            relations[index] = -1
    edges_a = list(filter(lambda x: x != -1, edges_a))
    edges_b = list(filter(lambda x: x != -1, edges_b))
    relations = list(filter(lambda x: x != -1, relations))
    graphs = dgl.graph((edges_a, edges_b))
    assert is_dgl_graph_connected(graphs), id
    assert len(node_sentences_a) + len(node_sentences_b) == graphs.num_nodes()

    if len(trans_exten_edge_ids) + len(cont_exten_node_ids) > Config.extension_padding_len:
        trans_exten_edge_ids = trans_exten_edge_ids[:Config.extension_padding_len]
        cont_exten_node_ids = cont_exten_node_ids[
                              :Config.extension_padding_len - len(trans_exten_edge_ids)]
    node_sentences_a = [sentence_format(s) for s in node_sentences_a]
    node_sentences_b = [sentence_format(s) for s in node_sentences_b]

    return context, answers, graphs, node_sentences_a, node_sentences_b, relations, get_edge_norm(relations, graphs,
                                                                                                  graphs.edges()), base_node_ids, cont_exten_node_ids, trans_exten_edge_ids
