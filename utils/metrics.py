import torchvision, torch
import numpy as np
import scipy
from sklearn.metrics import precision_recall_curve, auc, precision_recall_fscore_support

def compute_matching_score(nodes_list, probs, edges_list, gt_list, class_list, class_dict):
    """
    Calculating matching F1 score / Precision-Recall curve.

    node_list: a list of mung.node.Node, representing predicted nodes
    probs: a numpy array of shape (len(node_list), number_of_classes), representing probability distributions for each node.
    edges_list: a list of triplet (node1 idx, node2 idx, edge score), no repetition.
    gt_list: a list of mung.node.Node, representing ground truth nodes
    curve: whether to return a precesion-recall curve
    """
    # Get prediction id2idx
    id2idx_pred = {node.id: idx for idx, node in enumerate(nodes_list)}

    # Get box matrix from nodes_list
    boxes = [] 
    for node in nodes_list:
        boxes.append((node.left, node.top, node.right, node.bottom))


    # Filter invalid classes in gt_list
    gt_list_ = [node for node in gt_list if node.class_name in class_list]

    # Get box matrix and prob matrix from gt_list
    boxes_g = []
    prob_matrix = []
    id2idx = {}
    for idx, node in enumerate(gt_list_):
        id2idx[node.id] = idx
        prob_matrix.append(probs[:, class_dict[node.class_name]])
        boxes_g.append((node.left, node.top, node.right, node.bottom))
    prob_matrix = np.stack(prob_matrix).transpose()

    # Computer pairwise iou
    box_matrix = torchvision.ops.box_iou(torch.tensor(boxes), torch.tensor(boxes_g)).numpy()

    # Construct and solve matching problem
    cost_matrix = np.multiply(box_matrix, prob_matrix)
    row_indices, col_indices = scipy.optimize.linear_sum_assignment(-cost_matrix)
    match_a = {row_idx:col_idx for row_idx, col_idx in zip(row_indices, col_indices) if cost_matrix[row_idx, col_idx] > 0.05}
    # match_b = {col_idx:row_idx for row_idx, col_idx in zip(row_indices, col_indices) if cost_matrix[row_idx, col_idx] > 0.05}

    edges_dict = {(id2idx_pred[e[0]], id2idx_pred[e[1]]): e[2] for e in edges_list}
    labels = []
    preds = []

    for i in range(len(nodes_list)):
        for j in range(i, len(nodes_list)):
            if i not in match_a or j not in match_a:
                labels.append(False)
            else:
                u, v = match_a[i], match_a[j]
                labels.append((
                    gt_list_[u].id in gt_list_[v].outlinks) or 
                    (gt_list_[v].id in gt_list_[u].outlinks)
                )
            preds.append(edges_dict.get((i, j), 0.0))
    precision, recall, F1, _ = precision_recall_fscore_support(labels, np.array(preds)>0.5)
    precisions, recalls, _ = precision_recall_curve(labels, preds)
    auc_score = auc(recalls, precisions)
    return auc_score, F1[1], precision[1], recall[1]
