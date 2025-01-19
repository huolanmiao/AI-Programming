"""
本文件我们给出进行自动微分的步骤
你可以将lab5的对应代码复制到这里
"""

from typing import List, Dict, Tuple
from basic_operator import Op, Value
from operator import add
from functools import reduce

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """
    给定一个节点列表，返回以这些节点结束的拓扑排序列表。
    一种简单的算法是对给定的节点进行后序深度优先搜索（DFS）遍历，
    根据输入边向后遍历。由于一个节点是在其所有前驱节点遍历后才被添加到排序中的，
    因此我们得到了一个拓扑排序。
    """
    ## 请于此填写你的代码
    visited = set()  # 用于记录已访问的节点
    topo_order = []  # 存储拓扑排序结果
    
    # 对每个输入节点进行DFS
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
        
    return topo_order
    


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited: #已经访问过的节点不再访问
        return
    visited.add(node)
    for child in node.inputs:  # 递归访问所有子节点
        topo_sort_dfs(child, visited, topo_order)
    topo_order.append(node)  # 在节点的所有子节点遍历完之后，添加到排序列表中
    

def compute_gradient_of_variables(output_tensor, out_grad):
    """
    对输出节点相对于 node_list 中的每个节点求梯度。
    将计算结果存储在每个 Variable 的 grad 字段中。
    """
    # map for 从节点到每个输出节点的梯度贡献列表
    node_to_output_grads_list = {}
    # 我们实际上是在对标量 reduce_sum(output_node) 
    # 而非向量 output_node 取导数。
    # 但这是损失函数的常见情况。
    node_to_output_grads_list[output_tensor] = [out_grad]

    # 根据我们要对其求梯度的 output_node，以逆拓扑排序遍历图。
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ## 请于此填写你的代码
    # 遍历逆拓扑排序
    for node in reverse_topo_order:
        # 由于逆拓扑排序，可以求和计算output到node的梯度
        total_grads = reduce(add, node_to_output_grads_list[node])
        node.grad = total_grads
        for i, child in enumerate(node.inputs):  
            # 正向是由child经op计算得到node，反向是由node反向传播得到每个child的梯度
            parent_grad = node.op.gradient_as_tuple(total_grads, node)[i]  
            if child not in node_to_output_grads_list:
                node_to_output_grads_list[child] = []
            node_to_output_grads_list[child].append(parent_grad)
    





