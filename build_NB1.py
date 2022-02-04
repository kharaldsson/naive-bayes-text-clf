import sys
import os
import re
import numpy as np
import NaiveBayes

import random
import time

"""
BERNOULLI NB MODEL
"""
start = time.time()


def run(train_in, test_in, prior_delta, cond_delta, model_file, sys_output):
    start = time.time()
    with open(train_in, 'r', encoding='utf8') as f:
        train_lines = f.readlines()
    with open(test_in, 'r', encoding='utf8') as f:
        test_lines = f.readlines()

    clf = NaiveBayes.Bernoulli(train_lines, test_lines, prior_delta, cond_delta)
    clf.fit()
    clf.save_model(model_file)
    y_pred_tr, y_probs_tr = clf.predict(clf.X_train, save='train')
    y_pred_ts, y_probs_ts = clf.predict(clf.X_test, save='test')
    clf.save_sys_output(sys_output)
    clf.classification_report()
    end = time.time()
    total_time = end - start
    total_mins = total_time / 60
    fname = 'q2/time_' + str(prior_delta) + '_' + str(cond_delta)
    with open(fname, 'w', encoding='utf8') as f:
        f.write("time (s) " + str(total_time) + " | time (m) " + str(total_mins))


if __name__ == "__main__":
    TEST = False
    if TEST:
        TRAIN_IN = '/Users/Karl/_UW_Compling/LING572/hw3/hw3/examples/train.vectors.txt'
        TEST_IN = '/Users/Karl/_UW_Compling/LING572/hw3/hw3/examples/test.vectors.txt'
        PRIOR_DELTA = float(0.0)
        COND_DELTA = float(0.5)
        MODEL_FILE = 'q2/model_file'
        SYS_OUTPUT = 'q2/sys_output'
    else:
        TRAIN_IN = sys.argv[1]
        TEST_IN = sys.argv[2]
        PRIOR_DELTA = float(sys.argv[3])
        COND_DELTA = float(sys.argv[4])
        MODEL_FILE = sys.argv[5]
        SYS_OUTPUT = sys.argv[6]
    run(TRAIN_IN, TEST_IN, PRIOR_DELTA, COND_DELTA, MODEL_FILE, SYS_OUTPUT)
