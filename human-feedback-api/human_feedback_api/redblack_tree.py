from human_feedback_api.models import SortTree, Comparison

# See https://en.wikipedia.org/wiki/Tree_rotation
# and https://github.com/headius/redblack
# and https://www.cs.auckland.ac.nz/software/AlgAnim/red_black.html

def _root(tree):
    while tree.parent:
        tree = tree.parent
    return tree

def _left_rotate(x):
    """ Does a left tree-rotation around node x. """
    y = x.right
    x.set_right(y.left)
    if x.parent:
        if x == x.parent.left:
            x.parent.set_left(y)
        else:
            x.parent.set_right(y)
    else:  # y becomes new root
        y.parent = None
        y.save()
    y.set_left(x)

def _right_rotate(x):
    """ Does a right tree-rotation around node x. """
    y = x.left
    x.set_left(y.right)
    if x.parent:
        if x == x.parent.left:
            x.parent.set_left(y)
        else:
            x.parent.set_right(y)
    else:  # y becomes new root
        y.parent = None
        y.save()
    y.set_right(x)

def rebalance_tree(new_node):
    """ Rebalances a redblack tree after adding a node. """
    node = new_node
    while node.parent and node.parent.is_red:
        # Note: Since the root to the tree is always black, node must have a grandparent.
        if node.parent == node.parent.parent.left:  # node's parent is a left-child
            uncle = node.parent.parent.right
            if uncle and uncle.is_red:
                node.parent.make_black()
                uncle.make_black()
                node.parent.parent.make_red()
                node = node.parent.parent
            else:
                if node == node.parent.right:  # Triangle-shape
                    node = node.parent
                    _left_rotate(node)
                node.parent.make_black()
                node.parent.parent.make_red()
                _right_rotate(node.parent.parent)
        else:  # node's parent is a right-child
            uncle = node.parent.parent.left
            if uncle and uncle.is_red:
                node.parent.make_black()
                uncle.make_black()
                node.parent.parent.make_red()
                node = node.parent.parent
            else:
                if node == node.parent.left:  # Triangle-shape
                    node = node.parent
                    _right_rotate(node)
                node.parent.make_black()
                node.parent.parent.make_red()
                _left_rotate(node.parent.parent)
    _root(node).make_black()

class NewNodeNeeded(Exception):
    def __init__(self, moving_left):
        self.on_the_left = moving_left

def move_clip_down(base_node, clip, comparison_response):
    """
    Moves a given clip to a lower node the the redblack search/sort tree.
    If possible it recursively moves the clip multiple steps down, but usually just goes one step.
    Raises NewNodeNeeded if it hits the bottom of the tree.
    """
    # Okay, this is a little bit weird. Sorry for the awkwardness.
    # When the user responds "left" they're saying that the left clip is BETTER
    # In the redblack tree, left-nodes are worse, and right-nodes are better.
    # Since clip1 is the left clip, we want to move it left in the tree if the user says "right".
    move_left = (comparison_response == "right")
    if move_left and base_node.left:
        comparisons = Comparison.objects.filter(tree_node=base_node.left, left_clip=clip).exclude(response=None)
        if comparisons:
            move_clip_down(base_node.left, clip, comparisons[0].response)
        else:
            print(base_node.left, "gains", clip, "as new pending clip!")
            base_node.left.pending_clips.add(clip)
    elif (not move_left) and base_node.right:
        comparisons = Comparison.objects.filter(tree_node=base_node.right, left_clip=clip).exclude(response=None)
        if comparisons:
            move_clip_down(base_node.right, clip, comparisons[0].response)
        else:
            print(base_node.right, "gains", clip, "as new pending clip!")
            base_node.right.pending_clips.add(clip)
    else:
        raise NewNodeNeeded(move_left)
