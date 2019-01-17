#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified TIRA evaluator for sense classification with additional scores, confusion matrices, and Brier score, but without output prototext file.
(Supplementary task of CoNLL 2016 Shared Task)
"""
import json
import sys
from scorer import evaluate_connectives, evaluate_argument_extractor, evaluate_sense
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

    print('Explicit connectives         : Precision %1.4f Recall %1.4f F1 %1.4f' % connective_cm.get_prf('yes'))
    print('Arg 1 extractor              : Precision %1.4f Recall %1.4f F1 %1.4f' % arg1_cm.get_prf('yes'))
    print('Arg 2 extractor              : Precision %1.4f Recall %1.4f F1 %1.4f' % arg2_cm.get_prf('yes'))
    print('Arg1 Arg2 extractor combined : Precision %1.4f Recall %1.4f F1 %1.4f' % rel_arg_cm.get_prf('yes'))
    print('Sense classification--------------')
    sense_cm.print_summary(with_prf=True, with_ssi=True)
    print('Overall parser performance --------------')
    precision, recall, f1 = sense_cm.compute_micro_average_f1()
    sensitivity, specificity, informedness = sense_cm.compute_micro_average_informedness()
    print('Precision %1.4f Recall %1.4f F1 %1.4f' % (precision, recall, f1))
    print('Sensitivity %1.4f Specificity %1.4f Informedness %1.4f' % (sensitivity, specificity, informedness))
    print('Confusion matrix --------------')
    #sense_cm.print_matrix()
    sense_cm.print_matrix_with_pr()
    return connective_cm, arg1_cm, arg2_cm, rel_arg_cm, sense_cm, precision, recall, f1, sensitivity, specificity, informedness


def main(args):
    input_dataset = args[1]
    input_run = args[2]

    gold_relations = [json.loads(x) for x in open('%s/relations.json' % input_dataset)]
    predicted_relations = [json.loads(x) for x in open('%s/output.json' % input_run)]
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

    print('Evaluation for all discourse relations')
    extra_evaluate(gold_relations, predicted_relations)

    print('Evaluation for explicit discourse relations only')
    explicit_gold_relations = [x for x in gold_relations if x['Type'] == 'Explicit']
    explicit_predicted_relations = [x for x in predicted_relations if x['Type'] == 'Explicit']
    extra_evaluate(explicit_gold_relations, explicit_predicted_relations)

    print('Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)')
    non_explicit_gold_relations = [x for x in gold_relations if x['Type'] != 'Explicit']
    non_explicit_predicted_relations = [x for x in predicted_relations if x['Type'] != 'Explicit']
    extra_evaluate(non_explicit_gold_relations, non_explicit_predicted_relations)


if __name__ == '__main__':
    main(sys.argv)

