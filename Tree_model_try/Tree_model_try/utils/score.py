#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith
import numpy as np
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    f1 = calculate_F1_scores(gold_labels, test_labels)
    return score, cm, f1


def print_confusion_matrix(cm):
    lines = []
    precision = []
    recall = []

    for i in range(len(cm)):
        precision.append(cm[i][i]/(sum(cm[i])))
        recall.append(cm[i][i]/(sum(np.array(cm)[:, i])))
    # print(precision, recall)


    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS, 'precision')
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)
    # print(cm)
    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|{:^11.4f}|".format(LABELS[i],
                                                                   *row, precision[i]))
        lines.append("-"*line_len)
    lines.append("|{:^11}|{:^11.4f}|{:^11.4f}|{:^11.4f}|{:^11.4f}|{:^11}|".format('recall',recall[0], recall[1], recall[2], recall[3], ''))
    print('\n'.join(lines))
    # precision, recall = np.array(precision), np.array(recall)
    # f1 = 2 * np.multiply(precision, recall) / (precision + recall+1e-6)
    # print(f1)
    # print('f1-score : {:.4f}'.format(sum(f1) / len(f1)))

def calculate_F1_scores(y_true, y_predicted):
    from sklearn.metrics import f1_score
    #print(y_true)
    #print(y_predicted)
    f1_macro = f1_score(y_true, y_predicted, average='macro')
    f1_classwise = f1_score(y_true, y_predicted, average=None, labels=['agree', 'disagree', 'discuss', 'unrelated'])

    resultstring = "F1 macro: {:.3f}".format(f1_macro*100) + "% \n"
    resultstring += "F1 agree: {:.3f}".format(f1_classwise[0]*100) + "% \n"
    resultstring += "F1 disagree: {:.3f}".format(f1_classwise[1]*100) + "% \n"
    resultstring += "F1 discuss: {:.3f}".format(f1_classwise[2]*100) + "% \n"
    resultstring += "F1 unrelated: {:.3f}".format(f1_classwise[3]*100) + "% \n"

    #print(resultstring)
    return resultstring

def report_score(actual,predicted):
    score, cm, f1 = score_submission(actual,predicted)
    best_score, _, _ = score_submission(actual,actual)

    print_confusion_matrix(cm)
    print(f1)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score


if __name__ == "__main__":
    actual = [0,0,3,0,1,1,0,3,3,1,2,3,1]
    predicted = [0,0,3,0,1,1,2,3,3,0, 3, 2, 1]

    report_score([LABELS[e] for e in actual],[LABELS[e] for e in predicted])