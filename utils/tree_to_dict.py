import numpy as np


def _base_leaf(prediction: int) -> dict:
    return {
        'reference': 'true',  # queries correspond to feature == True on binarized data
        'relation': '==',
        'prediction': int(prediction)
    }


def _base_internal(feature_index: int) -> dict:
    return {
        'reference': 'true',
        'relation': '==',
        'feature': int(feature_index)
    }


def sklearn_tree_to_dict(tree) -> dict:
    """
    Convert sklearn.tree._tree.Tree into our dictionary schema.
    Assumes data are binarized so the right child corresponds to feature == 1 ("true").
    """
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold
    values = tree.value  # shape (n_nodes, 1, n_classes)

    def _recurse(node_id: int) -> dict:
        # Leaf: children are -1 in sklearn
        if children_left[node_id] == -1 and children_right[node_id] == -1:
            # pick class with max count
            class_counts = values[node_id][0]
            pred = int(np.argmax(class_counts))
            return _base_leaf(pred)

        # Internal node
        feat = int(features[node_id])
        node = _base_internal(feat)
        left = children_left[node_id]
        right = children_right[node_id]

        # For binarized features (0/1) with threshold ~0.5, sklearn routes:
        # x[feat] <= thresh -> left (feature == 0) and > thresh -> right (feature == 1)
        # Map right to 'true' and left to 'false'
        node['true'] = _recurse(right)
        node['false'] = _recurse(left)
        return node

    return _recurse(0)


def dl85_to_dict(node) -> dict:
    """
    Convert a pydl85 node dict into our dictionary schema.
    Expected keys:
      - internal: {'feat': int, 'left': node, 'right': node}
      - leaf: {'value': int}
    Left is feature == 1 in DL8.5? For binarized setup, we map
    left -> 'true' and right -> 'false' if left corresponds to positive branch.
    If project's convention is opposite, swap below accordingly.
    Here we follow introduce_mcar_rset_index.py which used left->true.
    """
    if 'value' in node:
        return _base_leaf(int(node['value']))

    out = _base_internal(int(node['feat']))
    out['true'] = dl85_to_dict(node['left'])
    out['false'] = dl85_to_dict(node['right'])
    return out


def gosdt_to_dict(gosdt_tree) -> dict:
    """
    Convert a GOSDT tree structure into our dictionary schema.
    gosdt_tree is expected to have attributes:
      - is_leaf (bool) or method is_leaf()
      - if leaf: label or prediction
      - if internal: variable (int) and true_branch/false_branch (or left/right)
    We support multiple common field names defensively.
    """
    # Determine API via duck-typing
    def is_leaf(node):
        if hasattr(node, 'is_leaf') and isinstance(node.is_leaf, bool):
            return node.is_leaf
        if hasattr(node, 'is_leaf') and callable(node.is_leaf):
            return node.is_leaf()
        if hasattr(node, 'leaf'):
            return bool(getattr(node, 'leaf'))
        # Fallback: presence of 'label' or 'prediction'
        return hasattr(node, 'label') or hasattr(node, 'prediction')

    def get_label(node):
        if hasattr(node, 'label'):
            return int(getattr(node, 'label'))
        if hasattr(node, 'prediction'):
            return int(getattr(node, 'prediction'))
        # Some GOSDT versions store distribution; take argmax if present
        if hasattr(node, 'counts'):
            return int(np.argmax(np.array(getattr(node, 'counts'))))
        raise AttributeError('Cannot find label for GOSDT leaf node')

    def get_feature(node):
        for attr in ['variable', 'feature', 'attribute', 'index']:
            if hasattr(node, attr):
                return int(getattr(node, attr))
        raise AttributeError('Cannot determine feature index for GOSDT internal node')

    def get_true_child(node):
        for attr in ['true', 'true_branch', 'right', 'right_child']:
            if hasattr(node, attr):
                return getattr(node, attr)
        raise AttributeError('Cannot find true/right child for GOSDT internal node')

    def get_false_child(node):
        for attr in ['false', 'false_branch', 'left', 'left_child']:
            if hasattr(node, attr):
                return getattr(node, attr)
        raise AttributeError('Cannot find false/left child for GOSDT internal node')

    def _recurse(node):
        if is_leaf(node):
            return _base_leaf(get_label(node))
        out = _base_internal(get_feature(node))
        out['true'] = _recurse(get_true_child(node))
        out['false'] = _recurse(get_false_child(node))
        return out

    return _recurse(gosdt_tree)


