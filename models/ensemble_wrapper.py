"""
Ensemble wrapper utilities for converting different tree formats to a common dictionary structure.
"""

import numpy as np


def sklearn_tree_to_dict(tree) -> dict:
    """
    Convert a sklearn DecisionTreeClassifier's internal tree structure to a dictionary format.
    
    Args:
        tree: sklearn.tree._tree.Tree object from a fitted DecisionTreeClassifier
        
    Returns:
        dict: Tree structure in the expected format for DNF tree classifiers
    """
    def _recurse_to_dict(node_id, source_node):
        node = {}
        node['id'] = node_id
        next_id = node_id + 1
        
        # Features needed for dnf_trees
        node['reference'] = 'true'  # queries assumed to always correspond to whether a feature is True
        node['relation'] = '=='
        
        # Check if this is a leaf node
        if source_node.children_left[node_id] == source_node.children_right[node_id]:  # leaf node
            # Get the prediction (class with highest probability)
            node['prediction'] = np.argmax(source_node.value[node_id][0])
        else:
            # Internal node
            node['feature'] = source_node.feature[node_id]
            node['threshold'] = source_node.threshold[node_id]
            
            # Recursively process left and right children
            left_id = source_node.children_left[node_id]
            right_id = source_node.children_right[node_id]
            
            if left_id == -1:  # left child is leaf
                node['false'] = {
                    "prediction": np.argmax(source_node.value[node_id][0]),
                    "id": next_id,
                    "reference": 'true',
                    "relation": '=='
                }
                next_id += 1
            else:
                node_false, next_id = _recurse_to_dict(next_id, source_node)
                node['false'] = node_false
            
            if right_id == -1:  # right child is leaf
                node['true'] = {
                    "prediction": np.argmax(source_node.value[node_id][0]),
                    "id": next_id,
                    "reference": 'true',
                    "relation": '=='
                }
                next_id += 1
            else:
                node_true, next_id = _recurse_to_dict(next_id, source_node)
                node['true'] = node_true
        
        return node, next_id
    
    return _recurse_to_dict(0, tree)[0]
