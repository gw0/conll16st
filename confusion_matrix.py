# -*- coding: utf-8 -*-
"""A collection of data structures that are particularly
useful for developing and improving a classifier
"""

import numpy
import json


class ConfusionMatrix(object):
    """Confusion matrix for evaluating a classifier

    For more information on confusion matrix en.wikipedia.org/wiki/Confusion_matrix
    """

    INIT_NUM_CLASSES = 100
    NEGATIVE_CLASS = '__NEGATIVE_CLASS__'
    def __init__(self, alphabet=None):
        if alphabet is None:
            self.alphabet = Alphabet()
            self.matrix = numpy.zeros((self.INIT_NUM_CLASSES, self.INIT_NUM_CLASSES))
        else:
            self.alphabet = alphabet
            num_classes = alphabet.size()
            self.matrix = numpy.zeros((num_classes,num_classes))

    def __iadd__(self, other):
        self.matrix += other.matrix
        return self

    def add(self, prediction, true_answer):
        """Add one data point to the confusion matrix

        If prediction is an integer, we assume that it's a legitimate index
        on the confusion matrix.

        If prediction is a string, then we will do the look up to
        map to the integer index for the confusion matrix.

        """
        if type(prediction) == int and type(true_answer) == int:
            self.matrix[prediction, true_answer] += 1
        else:
            self.alphabet.add(prediction)
            self.alphabet.add(true_answer)
            prediction_index = self.alphabet.get_index(prediction)
            true_answer_index = self.alphabet.get_index(true_answer)
            self.matrix[prediction_index, true_answer_index] += 1
            #XXX: this will fail if the prediction_index is greater than
            # the initial capacity. I should grow the matrix if this crashes


    def add_list(self, predictions, true_answers):
        """Add a list of data point to the confusion matrix

        A list can be a list of integers.
        If prediction is an integer, we assume that it's a legitimate index
        on the confusion matrix.

        A list can be a list of strings.
        If prediction is a string, then we will do the look up to
        map to the integer index for the confusion matrix.

        """
        for p, t in zip(predictions, true_answers):
            self.add(p, t)

    def get_prf_for_i(self, i):
        """Compute precision, recall, and f1 score for a given index."""

        if sum(self.matrix[i,:]) == 0:
            precision = 1.0
        else:
            precision = self.matrix[i,i] / sum(self.matrix[i,:])
        if sum(self.matrix[:,i]) == 0:
            recall = 1.0
        else:
            recall = self.matrix[i,i] / sum(self.matrix[:,i])
        if precision + recall != 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        return (precision, recall, f1)

    def get_ssi_for_i(self, i):
        """Compute sensitivity/recall, specificity, and informedness/Youden statistic."""
        tp = self.matrix[i,i]
        tn = sum(sum(self.matrix)) - sum(self.matrix[i,:]) - sum(self.matrix[:,i]) + self.matrix[i,i]
        fp = sum(self.matrix[i,:]) - self.matrix[i,i]
        fn = sum(self.matrix[:,i]) - self.matrix[i,i]
        if (tp + fn) == 0:
            sensitivity = 1.0
        else:
            sensitivity = tp / (tp + fn)  #= TP / (TP + FN)
        if (tn + fp) == 0:
            specificity = 1.0
        else:
            specificity = tn / (tn + fp)  #= TN / (TN + FP)
        informedness = sensitivity + specificity - 1.0
        return sensitivity, specificity, informedness

    def get_prf_for_all(self):
        """Compute precision, recall, and f1 score for all indexes."""

        precision = numpy.zeros(self.alphabet.size())
        recall = numpy.zeros(self.alphabet.size())
        f1 = numpy.zeros(self.alphabet.size())

        # compute precision, recall, and f1
        for i in xrange(self.alphabet.size()):
            precision[i], recall[i], f1[i] = self.get_prf_for_i(i)

        return (precision, recall, f1)

    def get_prf(self, class_name):
        """Compute precision, recall, and f1 score for a given class. """
        i = self.alphabet.get_index(class_name)
        return self.get_prf_for_i(i)

    def compute_micro_average_f1(self):
        """Compute micro-averaged precision, recall, and F1 score."""
        total_correct = 0.0
        for i in xrange(self.alphabet.size()):
            total_correct += self.matrix[i,i]
        negative_index = self.alphabet.get_index(self.NEGATIVE_CLASS)
        total_predicted = numpy.sum([x for i, x in enumerate(self.matrix.sum(1))\
            if negative_index == -1 or i != negative_index])
        total_gold = numpy.sum([x for i, x in enumerate(self.matrix.sum(0)) \
            if negative_index == -1 or i != negative_index])

        if total_predicted == 0:
            precision = 1.0
        else:
            precision = total_correct / total_predicted
        if total_gold == 0:
            recall = 1.0
        else:
            recall = total_correct / total_gold
        if precision + recall != 0.0:
            f1_score = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        return (round(precision, 4), round(recall, 4), round(f1_score,4))

    def compute_micro_average_informedness(self):
        """Compute micro-averaged sensitivity/recall, specificity, and informedness/Youden statistic."""
        tp = tn = fp = fn = 0
        for i in xrange(self.alphabet.size()):
            tp += self.matrix[i,i]
            tn += sum(sum(self.matrix)) - sum(self.matrix[i,:]) - sum(self.matrix[:,i]) + self.matrix[i,i]
            fp += sum(self.matrix[i,:]) - self.matrix[i,i]
            fn += sum(self.matrix[:,i]) - self.matrix[i,i]

        if (tp + fn) == 0:
            sensitivity = 1.0
        else:
            sensitivity = tp / (tp + fn)  #= TP / (TP + FN)
        if (tn + fp) == 0:
            specificity = 1.0
        else:
            specificity = tn / (tn + fp)  #= TN / (TN + FP)
        informedness = sensitivity + specificity - 1.0
        return sensitivity, specificity, informedness

    def compute_average_f1(self):
        precision, recall, f1 = self.get_prf_for_all()
        return numpy.mean(f1)

    def compute_average_prf(self):
        precision, recall, f1 = self.get_prf_for_all()
        return (round(numpy.mean(precision), 4),
                round(numpy.mean(recall), 4),
                round(numpy.mean(f1), 4))

    def print_matrix(self, mark_diag='*'):
        """Print confusion matrix for sense labels."""
        num_classes = self.alphabet.size()
        #header for the confusion matrix
        header = [' '] + [self.alphabet.get_label(i) for i in xrange(num_classes)]
        #putting labels to the first column of rhw matrix
        rows = []
        for i in xrange(num_classes):
            row = [self.alphabet.get_label(i)] + [str(self.matrix[i,j]) + (mark_diag if i == j else '') for j in xrange(num_classes)]
            rows.append(row)
        print("row = predicted, column = truth")
        print(matrix_to_string(rows, header))

    def print_matrix_with_pr(self, mark_diag='*'):
        """Print confusion matrix with precision and recall for sense labels."""
        num_classes = self.alphabet.size()
        precisions, recalls, _ = self.get_prf_for_all()
        total_true = numpy.sum(self.matrix, axis=0)
        total_pred = numpy.sum(self.matrix, axis=1)
        #header for the confusion matrix
        header = [' '] + [self.alphabet.get_label(i) for i in xrange(num_classes)] + ['__TOTAL__', '__PRECISION__']
        #putting labels to the first column of rhw matrix
        rows = []
        for i in xrange(num_classes):
            row = [self.alphabet.get_label(i)] + ["%d" % (self.matrix[i,j],) + (mark_diag if i == j else '') for j in xrange(num_classes)] + ["%d" % (total_pred[i],), "%1.4f" % (precisions[i],)]
            rows.append(row)
        row = ['__TOTAL__'] + ["%d" % (total_true[i],) for i in xrange(num_classes)] + ['', '']
        rows.append(row)
        row = ['__RECALL__'] + ["%1.4f" % (recalls[i],) for i in xrange(num_classes)] + ['', '']
        rows.append(row)
        print("row = predicted, column = truth")
        print(matrix_to_string(rows, header))

    def print_summary(self, with_prf=True, with_ssi=False):
        """Print summary of with PRF and/or SSI."""
        precision = numpy.zeros(self.alphabet.size())
        recall = numpy.zeros(self.alphabet.size())
        f1 = numpy.zeros(self.alphabet.size())
        sensitivity = numpy.zeros(self.alphabet.size())
        specificity = numpy.zeros(self.alphabet.size())
        informedness = numpy.zeros(self.alphabet.size())

        max_len = 0
        for i in xrange(self.alphabet.size()):
            label = self.alphabet.get_label(i)
            if label != self.NEGATIVE_CLASS and len(label) > max_len:
                max_len = len(label)

        lines = []
        correct = 0.0
        # prepare summary lines
        for i in xrange(self.alphabet.size()):
            precision[i], recall[i], f1[i] = self.get_prf_for_i(i)
            sensitivity[i], specificity[i], informedness[i] = self.get_ssi_for_i(i)
            correct += self.matrix[i,i]
            label = self.alphabet.get_label(i)
            if label != self.NEGATIVE_CLASS:
                space = ' ' * (max_len - len(label) + 1)
                line = '%s%s ' % (label, space)
                if with_prf:  # print precision, recall, and f1
                    line += 'precision %1.4f\trecall %1.4f\tF1 %1.4f' % (precision[i], recall[i], f1[i])
                if with_ssi:  # print sensitivity, specificity, informedness
                    line += '\tsensitivity %1.4f\tspecificity %1.4f\tinformedness %1.4f' % (sensitivity[i], specificity[i], informedness[i])
                lines.append(line)
        precision, recall, f1 = self.compute_micro_average_f1()
        sensitivity, specificity, informedness = self.compute_micro_average_informedness()
        label = '*Micro-Average'
        space = ' ' * (max_len - len(label) + 1)
        line = '%s%s ' % (label, space)
        if with_prf:  # print precision, recall, and f1
            line += 'precision %1.4f\trecall %1.4f\tF1 %1.4f' % (precision, recall, f1)
        if with_ssi:  # print sensitivity, specificity, informedness
            line += '\tsensitivity %1.4f\tspecificity %1.4f\tinformedness %1.4f' % (sensitivity, specificity, informedness)
        lines.append(line)
        #lines.sort()
        print('\n'.join(lines))

    def print_out(self):
        """Printing out confusion matrix along with Macro-F1 score"""
        self.print_matrix()
        self.print_summary()

    def plot(self, pdfname, normalize_axis=0, cmap=None):
        """Plot confusion matrix to PDF.

          pip install matplotlib
          apt-get install python-tk
        """
        import matplotlib
        matplotlib.use('Agg')  # do not use any Xwindows backend
        import matplotlib.pyplot as plt

        labels = [ self.alphabet.get_label(i)  for i in range(self.alphabet.size()) ]
        if normalize_axis is None:
            cm = self.matrix
        else:
            true_sums = numpy.sum(self.matrix, axis=normalize_axis, keepdims=True)
            cm = numpy.divide(self.matrix, true_sums, out=numpy.zeros_like(self.matrix), where=(true_sums != 0.))

        plt.figure(figsize=(0.65 * len(labels), 0.65 * len(labels)))
        plt.imshow(cm, interpolation="nearest", cmap=cmap or plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = numpy.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=90)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel("Predicted label")
        plt.xlabel("True label")

        plt.savefig(pdfname, bbox_inches='tight')


def matrix_to_string(matrix, header=None):
    """
    Return a pretty, aligned string representation of a nxm matrix.

    This representation can be used to print any tabular data, such as
    database results. It works by scanning the lengths of each element
    in each column, and determining the format string dynamically.

    the implementation is adapted from here
    mybravenewworld.wordpress.com/2010/09/19/print-tabular-data-nicely-using-python/

    Args:
        matrix - Matrix representation (list with n rows of m elements).
        header -  Optional tuple or list with header elements to be displayed.

    Returns:
        nicely formatted matrix string
    """

    if isinstance(header, list):
        header = tuple(header)
    lengths = []
    if header:
        lengths = [len(column) for column in header]

    #finding the max length of each column
    for row in matrix:
        for column in row:
            i = row.index(column)
            column = str(column)
            column_length = len(column)
            try:
                max_length = lengths[i]
                if column_length > max_length:
                    lengths[i] = column_length
            except IndexError:
                lengths.append(column_length)

    #use the lengths to derive a formatting string
    lengths = tuple(lengths)
    format_string = ""
    for length in lengths:
        format_string += "%-" + str(length) + "s "
    format_string += "\n"

    #applying formatting string to get matrix string
    matrix_str = ""
    if header:
        matrix_str += format_string % header
    for row in matrix:
        matrix_str += format_string % tuple(row)

    return matrix_str


class Alphabet(object):
    """Two way map for label and label index

    It is an essentially a code book for labels or features
    This class makes it convenient for us to use numpy.array
    instead of dictionary because it allows us to use index instead of
    label string. The implemention of classifiers uses label index space
    instead of label string space.
    """
    def __init__(self):
        self._index_to_label = {}
        self._label_to_index = {}
        self.num_labels = 0
        self.growing = True

    def __len__(self):
        return self.size()

    def __eq__(self, other):
        return self._index_to_label == other._index_to_label and \
            self._label_to_index == other._label_to_index and \
            self.num_labels == other.num_labels

    def size(self):
        return self.num_labels

    def has_label(self, label):
        return label in self._label_to_index

    def get_label(self, index):
        """Get label from index"""
        if index >= self.num_labels:
            raise KeyError("There are %d labels but the index is %d" % (self.num_labels, index))
        return self._index_to_label[index]

    def get_index(self, label):
        """Get index from label"""
        if not self.has_label(label):
            if self.growing:
                self.add(label) 
            else:
                return -1
        return self._label_to_index[label]

    def add(self, label):
        """Add an index for the label if it's a new label"""
        if label not in self._label_to_index:
            if not self.growing:
                raise ValueError(
                    'Alphabet is not set to grow i.e. accepting new labels')
            self._label_to_index[label] = self.num_labels
            self._index_to_label[self.num_labels] = label
            self.num_labels += 1

    def json_dumps(self):
        return json.dumps(self.to_dict())

    @classmethod
    def json_loads(cls, json_string):
        json_dict = json.loads(json_string)
        return Alphabet.from_dict(json_dict)

    def to_dict(self):
        return {
            '_label_to_index': self._label_to_index
            }

    @classmethod
    def from_dict(cls, alphabet_dictionary):
        """Create an Alphabet from dictionary

        alphabet_dictionary is a dictionary with only one field
        _label_to_index which is a map from label to index
        and should be created with to_dict method above.
        """
        alphabet = cls()
        alphabet._label_to_index = alphabet_dictionary['_label_to_index']
        alphabet._index_to_label = {}
        for label, index in alphabet._label_to_index.items():
            alphabet._index_to_label[index] = label
        # making sure that the dimension agrees
        assert(len(alphabet._index_to_label) == len(alphabet._label_to_index))
        alphabet.num_labels = len(alphabet._index_to_label)
        return alphabet


