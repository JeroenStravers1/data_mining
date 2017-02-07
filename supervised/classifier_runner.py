
"""
This script is meant for assessment purposes. It runs the two supervised learning algorithms (Naive Bayes & ID3) so
my grader can compare their accuracies.
"""

import supervised.decision_tree.decision_tree_classifier as dtc
import supervised.naive_bayes as nb


if __name__ == "__main__":

    decision_tree = dtc.ID3DecisionTree()

    # contains 8 "unseen" feature combinations
    decision_tree.train("../generated_sets/2017-01-13T13.55.08.756917_train_4911.csv")
    decision_tree.test("../generated_sets/2017-01-13T13.55.08.756917_test_4911.csv")

    # 1.0 accuracy
    # decision_tree.train("../generated_sets/2017-02-07T09.46.35.650308_train_9389.csv")
    # decision_tree.test("../generated_sets/2017-02-07T09.46.35.650308_test_9389.csv")

    # again, 1.0 accuracy
    # decision_tree.train("../generated_sets/2017-02-07T10.31.45.714731_train_8176.csv")
    # decision_tree.test("../generated_sets/2017-02-07T10.31.45.714731_test_8176.csv")
