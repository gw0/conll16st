#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified TIRA evaluator for sense classification with additional scores, confusion matrices, and Brier score, but without output prototext file.
(Supplementary task of CoNLL 2016 Shared Task)
"""
import json
import sys
from scorer import evaluate_connectives, evaluate_argument_extractor, evaluate_sense, spans_exact_matching, _link_gold_predicted
from validator import validate_relation_list, identify_language
from tira_eval import write_proto_text, write_results

def use_gold_standard_types(sorted_gold_relations, sorted_predicted_relations):
    for gr, pr in zip(sorted_gold_relations, sorted_predicted_relations):
        if gr['ID'] != pr['ID']:
            print('ID mismatch. Make sure you copy the ID from gold standard')
            exit(1)
        pr['Type'] = gr['Type']


def extra_evaluate(gold_list, predicted_list):
    connective_cm = evaluate_connectives(gold_list, predicted_list)
    arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, predicted_list)
    sense_cm = evaluate_sense(gold_list, predicted_list)
    brier_one, brier_all = evaluate_sense_proba(gold_list, predicted_list)

    print('Argument extraction --------------')
    print('Explicit connectives         : Precision %1.4f Recall %1.4f F1 %1.4f' % connective_cm.get_prf('yes'))
    print('Arg 1 extractor              : Precision %1.4f Recall %1.4f F1 %1.4f' % arg1_cm.get_prf('yes'))
    print('Arg 2 extractor              : Precision %1.4f Recall %1.4f F1 %1.4f' % arg2_cm.get_prf('yes'))
    print('Arg1 Arg2 extractor combined : Precision %1.4f Recall %1.4f F1 %1.4f' % rel_arg_cm.get_prf('yes'))

    print('Sense classification --------------')
    sense_cm.print_summary(with_prf=True, with_ssi=True)
    print('Confusion matrix --------------')
    #sense_cm.print_matrix()
    sense_cm.print_matrix_with_pr()

    print('Overall parser performance --------------')
    precision, recall, f1 = sense_cm.compute_micro_average_f1()
    sensitivity, specificity, informedness = sense_cm.compute_micro_average_informedness()
    print('Precision %1.4f Recall %1.4f F1 %1.4f' % (precision, recall, f1))
    print('Sensitivity %1.4f Specificity %1.4f Informedness %1.4f' % (sensitivity, specificity, informedness))
    print('Brier score %1.4f (%1.4f)' % (brier_one, brier_all))

    return connective_cm, arg1_cm, arg2_cm, rel_arg_cm, sense_cm, precision, recall, f1, sensitivity, specificity, informedness, brier_one, brier_all


def evaluate_sense_proba(gold_list, predicted_list):
    """
    Evaluate sense probabilities with Brier score.

    Example format of sense probabilities in `predicted_list`:
      predicted_list[i] = {
          "SenseProba": {"Causation": 0.03198, "Conditional": 0.0011, "Conjunction": 0.5055, ...},
      }

    - https://en.wikipedia.org/wiki/Brier_score
    """

    gold_to_predicted_map, predicted_to_gold_map =  _link_gold_predicted(gold_list, predicted_list, spans_exact_matching)

    one_sum = 0.
    all_sum = 0.
    n = 0
    for i, gold_relation in enumerate(gold_list):
        if i in gold_to_predicted_map:  # match
            predicted_relation = gold_to_predicted_map[i]

            # use only valid sense labels
            gold_senses = []
            for t in gold_relation['Sense']:
                if t in predicted_relation['SenseProba'].keys():
                    gold_senses.append(t)
            if len(gold_senses) == 0:
                print("!! skip: gold sense not in alphabet ({}, {})".format(gold_relation['ID'], gold_relation['Sense']))
                continue

            # use all gold sense labels (even if there are multiple)
            for t, p in predicted_relation['SenseProba'].items():
                if t in gold_senses:
                    all_sum += (p - 1.) ** 2
                else:
                    all_sum += (p - 0.) ** 2

            # use only one gold sense label (with highest predicted probability)
            gold_senses.sort(key=lambda t: predicted_relation['SenseProba'][t], reverse=True)
            gold_senses = gold_senses[:1]
            for t, p in predicted_relation['SenseProba'].items():
                if t in gold_senses:
                    one_sum += (p - 1.) ** 2
                else:
                    one_sum += (p - 0.) ** 2
            n += 1

        else:
            print("!! skip: no match for gold relation ({}, {})".format(gold_relation['ID'], gold_relation['Sense']))

    for i, predicted_relation in enumerate(predicted_list):
        if i not in predicted_to_gold_map:
            print("!! skip: no match for predicted relation ({}, {})".format(predicted_relation['ID'], predicted_relation['Sense']))

    return one_sum / n, all_sum / n


def main(args):
    input_dataset = args[1]
    input_run = args[2]

    gold_relations = [json.loads(x) for x in open('%s/relations.json' % input_dataset)]
    predicted_relations = [json.loads(x) for x in open('%s/output_proba.json' % input_run)]
    if len(gold_relations) != len(predicted_relations):
        err_message = 'Gold standard has % instances; predicted %s instances' % \
                (len(gold_relations), len(predicted_relations))
        print(err_message)
        exit(1)

    language = identify_language(gold_relations)
    all_correct = validate_relation_list(predicted_relations, language)
    if not all_correct:
        print('Invalid format')
        exit(1)

    gold_relations = sorted(gold_relations, key=lambda x: x['ID'])
    predicted_relations = sorted(predicted_relations, key=lambda x: x['ID'])
    use_gold_standard_types(gold_relations, predicted_relations)

    print('\nEvaluation for all discourse relations')
    extra_evaluate(gold_relations, predicted_relations)

    print('\nEvaluation for explicit discourse relations only')
    explicit_gold_relations = [x for x in gold_relations if x['Type'] == 'Explicit']
    explicit_predicted_relations = [x for x in predicted_relations if x['Type'] == 'Explicit']
    extra_evaluate(explicit_gold_relations, explicit_predicted_relations)

    print('\nEvaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)')
    non_explicit_gold_relations = [x for x in gold_relations if x['Type'] != 'Explicit']
    non_explicit_predicted_relations = [x for x in predicted_relations if x['Type'] != 'Explicit']
    extra_evaluate(non_explicit_gold_relations, non_explicit_predicted_relations)


if __name__ == '__main__':
    main(sys.argv)

