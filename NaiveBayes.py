import os
import re
import numpy as np
import time
from collections import Counter, OrderedDict
import math


# Superclass
class Classifier:
    def __init__(self, train_raw, test_raw, class_prior_delta, cond_prob_delta):
        self.train_raw = train_raw
        self.test_raw = test_raw
        self.class_prior_delta = class_prior_delta
        self.cond_prob_delta = cond_prob_delta
        self.n_classes = None
        self.n_docs = None
        self.n_docs_in_class = None
        self.vocab = None  # Set
        self.vocab_size = None  # int
        self.n_docs = None
        # self.classes = None
        self.vocab2idx = {}
        self.class2idx = {}
        self.idx2vocab = {}
        self.idx2class = {}
        self.prior_probs = None
        self.prior_lg_probs = None
        self.cond_probs = None
        self.cond_lg_probs = None
        self.y_hat_train = None
        self.y_hat_train_probs = None
        self.y_hat_test = None
        self.y_hat_test_probs = None

    @staticmethod
    def create_array(rows, columns):
        array_out = np.zeros((rows, columns))
        return array_out

    @staticmethod
    def prep_documents(data, type):
        data_clean = [re.sub('\n', '', x) for x in data]
        data_clean = [x for x in data_clean if x]
        data_clean = [re.split(r"\s+", x) for x in data_clean]

        # print(data_clean)

        # Split X and y
        y_str = [x[0] for x in data_clean]
        X_str = [x[1:] for x in data_clean]

        if type == 'multinomial':
            X_str = [[sl for sl in l if sl] for l in X_str]
            X_str = [[tuple(re.split(r":", sl)) for sl in l] for l in X_str]
            X_str = [dict(l) for l in X_str]
            X_str = [dict((k, int(v)) for k, v in subdict.items()) for subdict in X_str]
        else:
            X_str = [[re.split(r":", sl)[0] for sl in l] for l in X_str]
            X_str = [[sl for sl in l if sl] for l in X_str]

        return X_str, y_str

    def confusion_matrix(self, y_actual, y_predicted):
        conf_matrix = np.zeros((len(self.idx2class), len(self.idx2class)))
        # print(conf_matrix)
        for actual, pred in zip(y_actual, y_predicted):
            # print(str(actual) +' '+ str(pred))
            conf_matrix[actual, pred] += 1
        return conf_matrix

    def get_acc(self, y_actual, y_predicted):
        actual = np.array(y_actual)
        pred = np.array(y_predicted)
        correct = (actual == pred)
        accuracy = correct.sum() / correct.size
        return accuracy

    def classification_report(self):
        output_lines = ['Confusion matrix for the training data:', 'row is the truth, column is the system output',
                        '\n']

        train_matrix = self.confusion_matrix(self.y_train, self.y_hat_train)

        train_acc = self.get_acc(self.y_train, self.y_hat_train)
        test_matrix = self.confusion_matrix(self.y_test, self.y_hat_test)
        test_acc = self.get_acc(self.y_test, self.y_hat_test)

        class_labels = list(self.idx2class.values())

        class_labels_join = ' '.join(class_labels)
        class_labels_join = "\t\t" + class_labels_join
        output_lines.append(class_labels_join)

        for key, value in self.idx2class.items():
            matrix_counts = train_matrix[key, :].tolist()
            matrix_counts = [str(int(x)) for x in matrix_counts]
            matrix_counts = ' '.join(matrix_counts)
            matrix_line = str(value) + ' ' + matrix_counts
            output_lines.append(matrix_line)

        output_lines.append('\n')
        output_lines.append("Training accuracy=" + str(train_acc))
        output_lines.append('\n')
        second_title = ['Confusion matrix for the test data:', 'row is the truth, column is the system output',
                        '\n']
        output_lines += second_title
        output_lines.append(class_labels_join)

        for key, value in self.idx2class.items():
            matrix_counts = test_matrix[key, :].tolist()
            matrix_counts = [str(int(x)) for x in matrix_counts]
            matrix_counts = ' '.join(matrix_counts)
            matrix_line = str(value) + ' ' + matrix_counts
            output_lines.append(matrix_line)

        output_lines.append('\n')
        output_lines.append("Test accuracy=" + str(test_acc))

        for line in output_lines:
            print(line)

    def save_model(self, model_path):
        output_lines = []
        prior_header = "%%%%% prior prob P(c) %%%%%"
        output_lines.append(prior_header)

        pc_lines = []

        for class_idx, class_prob in enumerate(self.prior_probs):
            pc_line = str(self.idx2class[class_idx]) + " " + str(class_prob) + " " + str(self.prior_lg_probs[class_idx])
            pc_lines.append(pc_line)

        output_lines += pc_lines

        conditional_str = '%%%%% conditional prob P(f|c)'
        output_lines.append(conditional_str + "  %%%%%")

        for class_idx, class_name in self.idx2class.items():
            class_lines = []
            class_header = conditional_str + " c=" + class_name + " %%%%%"
            class_lines.append(class_header)
            class_lines_sorted = []
            for vocab_idx, vocab_prob in enumerate(self.cond_probs[class_idx, :]):
                vocab_name = self.idx2vocab[vocab_idx]
                prob = str(vocab_prob)
                lg_prob = str(self.cond_lg_probs[class_idx, vocab_idx])
                class_prob_line = str(vocab_name) + ' ' + str(class_name) + ' ' + prob + ' ' + lg_prob
                class_lines_sorted.append(class_prob_line)

            class_lines_sorted.sort()
            class_lines += class_lines_sorted

            output_lines += class_lines

        with open(model_path, 'w', encoding='utf8') as f:
            f.writelines("%s\n" % line for line in output_lines)

    def _get_proba_dist(self, cond_probs):
        # print(cond_probs)
        # cond_probs = np.array([-200.0, -201.0, -202.0])
        # print(cond_probs)
        exp = -np.max(cond_probs)
        probs = np.add(cond_probs, exp)
        # print(probs)
        probs_raised = np.power(10, probs)
        # print(probs_raised)
        denom = np.sum(probs_raised)
        # print(denom)
        new_probs = np.divide(probs_raised, denom)
        # print(new_probs)
        return new_probs

    def format_output_lines(self, predictions, set_header, line_header, actuals):
        all_lines = [set_header]
        # print(predictions)
        for doc_idx, pred_list in enumerate(predictions):

            normed = self._get_proba_dist(pred_list)
            # print(normed)
            line = []
            line_dict = {}
            l_header = line_header + str(doc_idx)
            l_actual = actuals[doc_idx]
            l_actual = self.idx2class[l_actual]
            line.append(l_header)
            line.append(l_actual)
            # print(self.idx2class.items())
            for class_idx, class_name in self.idx2class.items():
                # print(class_name)
                line_dict[class_name] = normed[class_idx]
            line_dict = OrderedDict(line_dict)
            for class_name, prob in sorted(line_dict.items(), key=lambda item: item[1], reverse=True):
                line_item = str(class_name) + " " + str(prob)
                line.append(line_item)

            line_string = ' '.join(line)
            all_lines.append(line_string)

        return all_lines

    def save_sys_output(self, sys_output_dir):
        """
        Write predictions to file
        """
        output_lines = []
        train_header = "%%%%% training data:"
        test_header = "%%%%% test data:"
        line_header = "array:"
        if self.y_hat_train is not None:
            y_hat_tr = self.y_hat_train_probs
            # output_lines.append(train_header)
            train_lines = self.format_output_lines(y_hat_tr, train_header, line_header, self.y_train)
            output_lines += train_lines

        if self.y_hat_test is not None:
            y_hat_ts = self.y_hat_test_probs
            output_lines.append('\n')
            # output_lines.append(test_header)
            test_lines = self.format_output_lines(y_hat_ts, test_header, line_header, self.y_test)
            output_lines += test_lines

        with open(sys_output_dir, 'w', encoding='utf8') as f:
            f.writelines("%s\n" % line for line in output_lines)

# Sublcass
class Multinomial(Classifier):
    def __init__(self, train_raw, test_raw, class_prior_delta, cond_prob_delta):
        self.X_train = None  # np array
        self.y_train = None  # np array
        self.X_test = None  # np array
        self.y_test = None  # np array
        self.class_unigram_counts = None
        self.class_keys = None
        super().__init__(train_raw, test_raw, class_prior_delta, cond_prob_delta)

        self.process_train()
        self.process_test()
        self._get_token_counts_per_class()

    def process_train(self):
        X_tr, y_tr = self.prep_documents(self.train_raw, type='multinomial')
        # print(X_tr)
        #
        # Set class information
        classes = list(dict.fromkeys(y_tr))  # set(y_tr)
        self.n_classes = len(classes)
        self.class2idx = {k: v for v, k in enumerate(classes)}
        self.idx2class = {k: v for k, v in enumerate(classes)}
        self.y_train = np.array([self.class2idx[c] for c in y_tr])
        class_count = dict(Counter(y_tr))
        self.n_docs_in_class = {self.class2idx[k]: v for k, v in class_count.items()}

        # Set number of docs
        self.n_docs = len(y_tr)

        # Set vocabulary
        self.vocab = [set(sub.keys()) for sub in X_tr]
        self.vocab = {item for sublist in self.vocab for item in sublist}
        self.vocab_size = len(self.vocab)
        vocabulary = list(self.vocab)
        # print(vocabulary)

        # Create vocabulary decoding dicts
        self.vocab2idx = {k: v for v, k in enumerate(vocabulary)}
        self.idx2vocab = {k: v for k, v in enumerate(vocabulary)}

        self.X_train = self.create_array(self.n_docs, self.vocab_size)
        self.class_unigram_counts = self.create_array(self.n_classes, self.vocab_size)
        for doc_idx, doc_dict in enumerate(X_tr):
            for word, word_count in doc_dict.items():
                word_idx = self.vocab2idx[word]
                self.X_train[doc_idx, word_idx] += word_count

    def process_test(self):
        X_ts, y_ts = self.prep_documents(self.test_raw, type='multinomial')

        # Set class information
        self.y_test = np.array([self.class2idx[c] for c in y_ts if c in self.class2idx])

        # Set number of docs
        n_test_docs = len(y_ts)

        self.X_test = self.create_array(n_test_docs, self.vocab_size)

        for doc_idx, doc_dict in enumerate(X_ts):
            for word, word_count in doc_dict.items():
                if word in self.vocab2idx:
                    word_idx = self.vocab2idx[word]
                    self.X_test[doc_idx, word_idx] += word_count

    def _get_token_counts_per_class(self):
        class_word_cnt = np.zeros(self.vocab_size)
        class_keys = sorted(self.idx2class.keys())
        self.class_keys = class_keys
        # class_word_cnt
        for class_idx in class_keys:
            doc_idxs = np.where(self.y_train == class_idx)[0]
            X_tr_docs = self.X_train[doc_idxs]
            count = np.sum(X_tr_docs, axis=0)
            class_word_cnt = np.vstack([class_word_cnt, count])
        class_word_cnt = np.delete(class_word_cnt, 0, 0)
        self.class_unigram_counts = class_word_cnt

    def fit(self):
        prior_probs = []

        for class_idx in self.class_keys:
            N_c = self.n_docs_in_class[class_idx]
            # print(N_c)
            class_prior_prob = (N_c + self.class_prior_delta) / (
                    self.n_docs + (self.class_prior_delta * self.n_classes))
            prior_probs.append(class_prior_prob)

        smoothed_wc = np.add(self.class_unigram_counts, self.cond_prob_delta)
        smoothed_cwc = smoothed_wc.sum(axis=1, keepdims=True)
        smoothed_denom = self.cond_prob_delta * self.vocab_size
        smoothed_cwc = np.add(smoothed_cwc, smoothed_denom)
        self.cond_probs = np.divide(smoothed_wc, smoothed_cwc)
        # self.cond_probs = smoothed_wc / (smoothed_wc.sum(axis=1, keepdims=True) + (self.vocab_size * self.cond_prob_delta))
        self.cond_lg_probs = np.log10(self.cond_probs)
        self.prior_probs = np.array(prior_probs)
        self.prior_lg_probs = np.log10(self.prior_probs)

    def predict(self, X_array, save=None):
        y_pred = np.zeros(np.shape(X_array)[0]).astype(int)
        y_probs = []

        for doc_idx, doc in enumerate(X_array):
            c_pred, c_probs = self.predict_proba(X_array[doc_idx, :])
            y_pred[doc_idx] = c_pred
            y_probs.append(c_probs)

        if save == 'test':
            self.y_hat_test = y_pred
            self.y_hat_test_probs = y_probs
        elif save == 'train':
            self.y_hat_train = y_pred
            self.y_hat_train_probs = y_probs

        return y_pred, y_probs

    def predict_proba(self, test_instance):
        probs_permut = np.multiply(test_instance, self.cond_lg_probs)
        probs_summed = np.sum(probs_permut, axis=1, keepdims=True).flatten()
        probs = np.add(probs_summed, self.prior_lg_probs)

        pred = np.argmax(probs)

        return pred, probs

# Sublcass
class Bernoulli(Classifier):
    def __init__(self, train_raw, test_raw, class_prior_delta, cond_prob_delta):
        self.X_train = None  # np array
        self.y_train = None  # np array
        self.X_test = None  # np array
        self.y_test = None  # np array
        self.n_docs_word_in_class = None
        self.cond_lg_probs_not = None
        self.cond_prob_ratios = None
        self.lg_probs_not_sum = None
        super().__init__(train_raw, test_raw, class_prior_delta, cond_prob_delta)

        self.process_train()
        self.process_test()
        self._get_cnt_docs_given_w_and_c()

    def process_train(self):
        X_tr, y_tr = self.prep_documents(self.train_raw, type='bernoulli')
        # print(X_tr[0:5])

        # Set class information
        classes = list(dict.fromkeys(y_tr))  # set(y_tr)
        self.n_classes = len(classes)
        self.class2idx = {k: v for v, k in enumerate(classes)}
        self.idx2class = {k: v for k, v in enumerate(classes)}
        self.y_train = np.array([self.class2idx[c] for c in y_tr])
        class_count = dict(Counter(y_tr))
        self.n_docs_in_class = {self.class2idx[k]: v for k, v in class_count.items()}

        # Set number of docs
        self.n_docs = len(y_tr)

        # Set vocabulary
        self.vocab = set([word for doc in X_tr for word in doc])
        self.vocab_size = len(self.vocab)
        vocabulary = list(self.vocab)

        # Create vocabulary decoding dicts
        self.vocab2idx = {k: v for v, k in enumerate(vocabulary)}
        self.idx2vocab = {k: v for k, v in enumerate(vocabulary)}

        X_array = [[self.vocab2idx[word] for word in doc] for doc in X_tr]
        self.X_train = self.create_array(self.n_docs, self.vocab_size)

        # print(_X)

        for doc_idx, doc in enumerate(X_array):
            for word in doc:
                self.X_train[doc_idx, word] = 1

        # print(X_array)
        # print(self.X_train)

    def process_test(self):
        X_ts, y_ts = self.prep_documents(self.test_raw, type='bernoulli')

        # Set class information
        self.y_test = np.array([self.class2idx[c] for c in y_ts if c in self.class2idx])

        # Set number of docs
        n_test_docs = len(y_ts)

        X_array = [[self.vocab2idx[word] for word in doc if word in self.vocab2idx] for doc in X_ts]
        self.X_test = self.create_array(n_test_docs, self.vocab_size)

        for doc_idx, doc in enumerate(X_array):
            for word in doc:
                self.X_test[doc_idx, word] = 1

    def _get_cnt_docs_given_w_and_c(self):
        class_word_cnt = np.zeros(self.vocab_size)
        # class_word_cnt
        for class_idx in self.idx2class.keys():
            doc_idxs = np.where(self.y_train == class_idx)[0]
            X_tr_docs = self.X_train[doc_idxs]
            count = np.count_nonzero(X_tr_docs, axis=0)
            class_word_cnt = np.vstack([class_word_cnt, count])
        class_word_cnt = np.delete(class_word_cnt, 0, 0)
        self.n_docs_word_in_class = class_word_cnt

    def fit(self):
        prior_delta = self.class_prior_delta
        cond_delta = self.cond_prob_delta
        # vocab = set(self.idx2vocab.keys())
        n_docs = self.n_docs
        self.cond_probs = np.zeros((self.n_classes, self.vocab_size))
        prior_probs = np.zeros(self.n_classes)
        # print(self.vocab_size)
        for c in self.idx2class.keys():
            # print(c)
            N_c = self.n_docs_in_class[c]
            prior = (N_c + prior_delta) / (n_docs + (prior_delta * self.n_classes))
            prior_probs[c] = prior
            for word_idx in self.idx2vocab.keys():
                N_ct = self.n_docs_word_in_class[c, word_idx]
                cond_prob = (N_ct + cond_delta) / (N_c + (2 * cond_delta))
                self.cond_probs[c, word_idx] = cond_prob
        self.prior_probs = prior_probs
        self.prior_lg_probs = np.log10(self.prior_probs)

        self.cond_lg_probs = np.log10(self.cond_probs)

        cond_probs_not = 1 - self.cond_probs
        self.cond_lg_probs_not = np.log10(cond_probs_not)

        self.cond_prob_ratios = np.subtract(self.cond_lg_probs, self.cond_lg_probs_not)

        Z_c = np.sum(self.cond_lg_probs_not, axis=1, keepdims=True)

        self.lg_probs_not_sum = Z_c

    def predict(self, X_array, save=None):
        y_pred = np.zeros(np.shape(X_array)[0]).astype(int)
        y_probs = []

        for doc_idx, doc in enumerate(X_array):
            c_pred, c_probs = self.predict_proba(X_array[doc_idx, :])
            y_pred[doc_idx] = c_pred
            y_probs.append(c_probs)

        if save == 'test':
            self.y_hat_test = y_pred
            self.y_hat_test_probs = y_probs
        elif save == 'train':
            self.y_hat_train = y_pred
            self.y_hat_train_probs = y_probs

        return y_pred, y_probs

    def predict_proba(self, test_instance):

        probs = np.zeros(self.n_classes)
        for c in self.idx2class.keys():
            class_prior_prob = self.prior_lg_probs[c]
            class_lg_prob_not = self.lg_probs_not_sum[c]
            lg_prob = class_prior_prob + class_lg_prob_not[0]

            test = np.multiply(self.cond_prob_ratios[c, :], test_instance)
            test = np.sum(test)
            test = np.add(test, lg_prob)
            probs[c] = test

        pred = np.argmax(probs).astype(int)

        return pred, probs