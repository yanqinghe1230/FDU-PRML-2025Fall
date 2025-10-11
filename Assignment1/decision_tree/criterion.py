"""
criterion
"""
import math

def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels

def entropy(labels,total):
    ent=0.0
    for cnt in labels:
        if cnt==0:
            continue 
        p=cnt/total
        ent-=p*math.log2(p)
    
    return ent
    
def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    info_gain = 0.0
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    total=len(y)
    total_l=len(left_labels)
    total_r=len(right_labels)
    
    ent_p=entropy(all_labels,total) if total>0 else 0.0
    ent_l=entropy(left_labels,total_l) if total_l>0 else 0.0
    ent_r=entropy(right_labels,total_r) if total_r>0 else 0.0

    info_gain=ent_p-(total_l/total)*ent_l-(total_r/total)*ent_r
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    total=len(y)
    total_l=len(l_y)
    total_r=len(r_y)
    
    split_info=0.0
    split_info-=total_l/total*math.log2(total_l/total) if total_l>0 else 0.0
    split_info-=total_r/total*math.log2(total_r/total) if total_r>0 else 0.0
    
    info_gain_ratio=info_gain/split_info if split_info!=0 else 0
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain_ratio

def gini(labels,total):
    gini_index=1.0
    for cnt in labels:
        p=cnt/total
        gini_index-=p**2
    return gini_index

def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    before=gini(all_labels,len(y)) if len(y)>0 else 0.0
    after_l=gini(left_labels,len(l_y)) if len(l_y)>0 else 0.0
    after_r=gini(right_labels,len(r_y)) if len(r_y)>0 else 0.0

    after=(len(l_y)/len(y))*after_l+(len(r_y)/len(y))*after_r
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after

def max_p(labels,total):
    max_p=0.0
    max_cnt=max(labels)
    max_p=max_cnt/total
    return max_p

def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    before=1-max_p(all_labels,len(y)) if len(y)>0 else 0.0
    after_l=1-max_p(left_labels,len(l_y)) if len(l_y)>0 else 0.0
    after_r=1-max_p(right_labels,len(r_y)) if len(r_y)>0 else 0.0
    after=(len(l_y)/len(y))*after_l+(len(r_y)/len(y))*after_r
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
