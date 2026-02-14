"""
===================================================================
RadixTree - 前缀匹配树 (用于KV Cache共享)
===================================================================

📌 **核心设计目标**：
   1. 高效存储和检索token序列
   2. 支持前缀匹配和公共前缀查找
   3. O(k) 时间复杂度查找 (k = 查询序列长度)
   4. 支持序列管理和复用

🧱 **数据结构**：
   每个节点代表一个token，包含：
   - token: 当前token值
   - children: 子节点字典 (token -> RadixNode)
   - seq_ids: 存储经过此节点的序列ID集合
   - is_end: 是否为序列结束节点
   - block_id: 对应的缓存块ID (用于KV Cache复用)

📊 **典型用途**：
   1. 前缀缓存复用 - 多个请求共享相同前缀时复用KV Cache
   2. 注意力机制优化 - 避免重复计算相同前缀的Attention
   3. 缓存淘汰策略 - 基于前缀热度进行LRU淘汰

⚡ **性能特性**：
   - 插入: O(k) where k is token count
   - 前缀查找: O(k) where k is prefix length
   - 公共前缀查找: O(min(k1, k2))
   - 空间: O(total_tokens) 每个token一个节点
"""
from typing import List, Optional, Set, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class RadixNode:
    """Radix树节点"""
    token: int  # token值
    children: Dict[int, 'RadixNode'] = field(default_factory=dict)  # 子节点
    seq_ids: Set[int] = field(default_factory=set)  # 经过此节点的序列ID
    is_end: bool = False  # 是否为序列结束节点
    block_id: Optional[int] = None  # 对应的缓存块ID
    seq_len: int = 0  # 从根节点到此节点的序列长度


class RadixTree:
    """
    📌 **Radix前缀树** - 用于KV Cache前缀匹配和复用
    
    🔍 **核心功能**：
       1. 插入序列 - 将新序列加入树中
       2. 前缀匹配 - 查找与给定序列有公共前缀的其他序列
       3. 序列复用 - 找到可复用的KV Cache块
       4. 前缀长度计算 - 计算公共前缀长度
       
    🧪 **典型用法**:
       tree = RadixTree(max_blocks=1024)
       
       # 插入序列
       tree.insert(seq_id=1, tokens=[1, 2, 3, 4, 5])
       tree.insert(seq_id=2, tokens=[1, 2, 3, 6, 7])
       
       # 查找公共前缀
       common = tree.find_longest_prefix(tokens=[1, 2, 3, 4, 5])
       # 返回: (matched_length=3, matched_seq_ids={1, 2})
       
       # 获取可复用的block
       blocks = tree.get_reusable_blocks(seq_id=1)
    """
    
    def __init__(self, max_blocks: int = 1024):
        """
        📌 **初始化**
        
        🔍 **参数**:
            - max_blocks: 最大缓存块数
        """
        # 根节点 (空token，表示序列开始)
        self._root = RadixNode(token=-1, seq_len=0)
        
        # 序列信息存储
        self._seq_info: Dict[int, List[int]] = {}  # seq_id -> tokens
        self._seq_blocks: Dict[int, List[int]] = {}  # seq_id -> block_ids
        
        # 统计信息
        self._total_sequences: int = 0
        self._max_blocks = max_blocks
        
        # 节点计数 (用于调试)
        self._node_count: int = 0
    
    def insert(self, seq_id: int, tokens: List[int], block_ids: Optional[List[int]] = None) -> bool:
        """
        📌 **插入序列**
        
        🔍 **参数**:
            - seq_id: 序列ID (唯一标识)
            - tokens: token列表
            - block_ids: 对应的缓存块ID列表 (可选)
            
        ✅ **返回**:
            - 是否插入成功
            
        🧠 **内部逻辑**:
            1. 如果序列已存在，先删除
            2. 从根节点开始遍历
            3. 对于每个token:
               - 如果子节点存在，沿路径继续
               - 否则创建新节点
            4. 在末尾节点标记is_end=True
            5. 记录序列信息和block映射
        """
        # 如果序列已存在，先删除
        if seq_id in self._seq_info:
            self.delete(seq_id)
        
        # 插入tokens
        node = self._root
        node.seq_ids.add(seq_id)  # 根节点也记录序列ID
        
        for i, token in enumerate(tokens):
            if token not in node.children:
                # 创建新节点
                new_node = RadixNode(token=token, seq_len=i+1)
                node.children[token] = new_node
                self._node_count += 1
            
            # 沿路径移动
            node = node.children[token]
            node.seq_ids.add(seq_id)  # 每个节点记录经过的序列ID
        
        # 标记序列结束
        node.is_end = True
        
        # 记录序列信息
        self._seq_info[seq_id] = tokens.copy()
        
        # 记录block映射
        if block_ids is None:
            # 自动生成block_ids
            n_blocks = (len(tokens) + 15) // 16  # 假设block_size=16
            block_ids = list(range(n_blocks))
        self._seq_blocks[seq_id] = block_ids.copy()
        
        self._total_sequences += 1
        return True
    
    def delete(self, seq_id: int) -> bool:
        """
        📌 **删除序列**
        
        🔍 **参数**:
            - seq_id: 序列ID
            
        ✅ **返回**:
            - 是否删除成功
        """
        if seq_id not in self._seq_info:
            return False
        
        tokens = self._seq_info[seq_id]
        
        # 从每个节点中移除seq_id
        node = self._root
        node.seq_ids.discard(seq_id)
        
        for token in tokens:
            if token in node.children:
                node = node.children[token]
                node.seq_ids.discard(seq_id)
            else:
                break  # 序列不存在
        
        # 清理不再需要的节点 (递归删除)
        self._cleanup_nodes(self._root, tokens, 0)
        
        # 删除序列信息
        del self._seq_info[seq_id]
        del self._seq_blocks[seq_id]
        self._total_sequences -= 1
        
        return True
    
    def _cleanup_nodes(self, node: RadixNode, tokens: List[int], depth: int):
        """递归清理不再被任何序列使用的节点"""
        if depth >= len(tokens):
            return
        
        token = tokens[depth]
        if token not in node.children:
            return
        
        child = node.children[token]
        self._cleanup_nodes(child, tokens, depth + 1)
        
        # 如果子节点不再被任何序列使用，且不是结束节点，则删除
        if not child.seq_ids and not child.is_end and not child.children:
            del node.children[token]
            self._node_count -= 1
    
    def find_longest_prefix(self, tokens: List[int]) -> Tuple[int, Set[int]]:
        """
        📌 **查找最长公共前缀**
        
        🔍 **参数**:
            - tokens: 待查询的token序列
            
        ✅ **返回**:
            - (matched_length, matched_seq_ids): 公共前缀长度和所有能匹配到这个前缀的序列ID集合
            
        🧠 **内部逻辑**:
            1. 从根节点开始遍历token序列
            2. 记录每个位置对应的seq_ids集合
            3. 找到seq_ids集合开始变化的位置，返回变化前的长度和序列ID
            4. 这表示所有能匹配到的序列的公共前缀
            
        📊 **示例**:
            - 序列1: [1,2,3,4,5], 序列2: [1,2,3,6,7]
            - 查询 [1,2,3,4,8]
            - 返回 (3, {1, 2}) - 公共前缀长度为3，序列1和2都能匹配
        """
        if not tokens:
            return 0, self._root.seq_ids.copy()
        
        node = self._root
        
        # 记录每个成功匹配位置的seq_ids
        seq_ids_at_pos = []
        
        for i, token in enumerate(tokens):
            if token in node.children:
                node = node.children[token]
                # 记录这个位置的seq_ids
                seq_ids_at_pos.append((i + 1, node.seq_ids.copy()))
            else:
                # token不匹配，停止
                break
        
        # 找到公共前缀：找到seq_ids开始变化的位置，返回变化前的长度和序列ID
        if len(seq_ids_at_pos) > 1:
            final_seq_ids = seq_ids_at_pos[-1][1]
            for j in range(len(seq_ids_at_pos) - 2, -1, -1):
                length, seq_ids = seq_ids_at_pos[j]
                if seq_ids != final_seq_ids:
                    # 找到变化点，返回前一个位置的信息 (j位置)
                    return seq_ids_at_pos[j]
            
            # 如果没有变化，返回最后一个
            return seq_ids_at_pos[-1]
        
        # 如果只有一个匹配位置或没有匹配
        if seq_ids_at_pos:
            return seq_ids_at_pos[0]
        
        # 如果没有匹配，返回根节点信息
        return 0, self._root.seq_ids.copy()
    
    def find_all_matching(self, tokens: List[int]) -> List[int]:
        """
        📌 **查找所有经过给定前缀的序列ID**
        
        🔍 **参数**:
            - tokens: 待查询的token序列
            
        ✅ **返回**:
            - 所有经过给定前缀的序列ID列表 (按seq_id排序)
            
        🧠 **内部逻辑**:
            遍历token序列直到遇到不匹配的节点，返回该节点及其之前所有节点的seq_ids的并集
        """
        node = self._root
        all_seq_ids = set()
        
        for i, token in enumerate(tokens):
            if token in node.children:
                node = node.children[token]
                # 累加所有经过的seq_ids
                all_seq_ids.update(node.seq_ids)
            else:
                # token不匹配，停止
                break
        
        return sorted(list(all_seq_ids))
    
    def get_reusable_blocks(self, seq_id: int) -> List[int]:
        """
        📌 **获取可复用的缓存块**
        
        🔍 **参数**:
            - seq_id: 当前序列ID
            
        ✅ **返回**:
            - 可复用的block_id列表
            
        🧠 **说明**:
            - 返回当前序列之前已存在的、可复用的block
            - 只返回完全匹配的block (整块复用)
        """
        if seq_id not in self._seq_info:
            return []
        
        current_tokens = self._seq_info[seq_id]
        
        # 查找最长前缀
        matched_len, matched_seqs = self.find_longest_prefix(current_tokens)
        
        if matched_len == 0 or not matched_seqs:
            return []
        
        # 排除当前序列自身
        matched_seqs.discard(seq_id)
        
        if not matched_seqs:
            return []
        
        # 找到匹配的序列，获取其block
        # matched_len 个token对应的block数量
        reusable_blocks = matched_len // 16  # 假设block_size=16
        
        # 从匹配的序列中获取可用的block
        result_blocks = []
        for matched_seq in matched_seqs:
            if matched_seq in self._seq_blocks:
                blocks = self._seq_blocks[matched_seq]
                result_blocks.extend(blocks[:reusable_blocks])
        
        return list(set(result_blocks))
    
    def get_prefix_blocks(self, prefix_tokens: List[int]) -> Tuple[int, List[int]]:
        """
        📌 **获取前缀对应的block信息**
        
        🔍 **参数**:
            - prefix_tokens: 前缀token序列
            
        ✅ **返回**:
            - (prefix_len, block_ids): 实际匹配的前缀长度和对应的block列表
        """
        matched_len, matched_seqs = self.find_longest_prefix(prefix_tokens)
        
        if matched_len == 0 or not matched_seqs:
            return 0, []
        
        # 计算匹配的block数量 (假设block_size=16)
        matched_blocks = matched_len // 16
        
        # 收集所有匹配序列的block
        all_blocks = []
        for seq_id in matched_seqs:
            if seq_id in self._seq_blocks:
                all_blocks.extend(self._seq_blocks[seq_id][:matched_blocks])
        
        return matched_len, list(set(all_blocks))
    
    def get_sequence(self, seq_id: int) -> Optional[List[int]]:
        """
        📌 **获取序列的tokens**
        
        🔍 **参数**:
            - seq_id: 序列ID
            
        ✅ **返回**:
            - token列表，如果不存在则返回None
        """
        return self._seq_info.get(seq_id)
    
    def has_prefix(self, tokens: List[int]) -> bool:
        """
        📌 **检查是否存在给定前缀的序列**
        
        🔍 **参数**:
            - tokens: 待检查的前缀
            
        ✅ **返回**:
            - 是否存在匹配的前缀
        """
        node = self._root
        for token in tokens:
            if token in node.children:
                node = node.children[token]
            else:
                return False
        return True
    
    def get_all_sequences(self) -> List[int]:
        """
        📌 **获取所有序列ID**
        
        ✅ **返回**:
            - 所有序列ID列表
        """
        return sorted(list(self._seq_info.keys()))
    
    def clear(self):
        """清空整棵树"""
        self._root = RadixNode(token=-1, seq_len=0)
        self._seq_info.clear()
        self._seq_blocks.clear()
        self._total_sequences = 0
        self._node_count = 0
    
    @property
    def stats(self) -> Dict:
        """
        📌 **获取统计信息**
        
        ✅ **返回**:
            - node_count: 节点数量
            - seq_count: 序列数量
            - max_blocks: 最大块数
        """
        return {
            "node_count": self._node_count,
            "seq_count": self._total_sequences,
            "max_blocks": self._max_blocks,
        }
    
    def __len__(self) -> int:
        """返回序列数量"""
        return self._total_sequences
    
    def __contains__(self, seq_id: int) -> bool:
        """检查序列是否存在"""
        return seq_id in self._seq_info


# =============================================================================
# 🧪 使用示例
# =============================================================================

if __name__ == "__main__":
    # 创建Radix树
    tree = RadixTree(max_blocks=1024)
    
    # 插入序列
    print("=== 插入序列 ===")
    tree.insert(seq_id=1, tokens=[1, 2, 3, 4, 5], block_ids=[0, 1])
    tree.insert(seq_id=2, tokens=[1, 2, 3, 6, 7], block_ids=[2, 3])
    tree.insert(seq_id=3, tokens=[1, 2, 3, 4, 5, 6, 7, 8], block_ids=[4, 5, 6])
    
    print(f"统计: {tree.stats}")
    print(f"所有序列: {tree.get_all_sequences()}")
    
    # 查找最长前缀
    print("\n=== 查找最长前缀 ===")
    matched_len, matched_seqs = tree.find_longest_prefix([1, 2, 3, 4, 5])
    print(f"查询 [1,2,3,4,5]: 匹配长度={matched_len}, 序列IDs={matched_seqs}")
    
    matched_len, matched_seqs = tree.find_longest_prefix([1, 2, 3, 9])
    print(f"查询 [1,2,3,9]: 匹配长度={matched_len}, 序列IDs={matched_seqs}")
    
    matched_len, matched_seqs = tree.find_longest_prefix([1, 2, 3, 4, 5, 6, 7])
    print(f"查询 [1,2,3,4,5,6,7]: 匹配长度={matched_len}, 序列IDs={matched_seqs}")
    
    # 检查前缀存在
    print("\n=== 检查前缀 ===")
    print(f"前缀 [1,2,3] 存在: {tree.has_prefix([1, 2, 3])}")
    print(f"前缀 [1,2,3,4,5,6] 存在: {tree.has_prefix([1, 2, 3, 4, 5, 6])}")
    print(f"前缀 [1,2,3,4,5,6,7,8,9] 存在: {tree.has_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9])}")
    
    # 获取可复用块
    print("\n=== 获取可复用块 ===")
    # 插入seq_id=4，与seq_id=1有公共前缀
    tree.insert(seq_id=4, tokens=[1, 2, 3, 4, 5, 9, 10])
    reusable = tree.get_reusable_blocks(seq_id=4)
    print(f"序列4 [1,2,3,4,5,9,10] 可复用块: {reusable}")
    
    # 删除序列
    print("\n=== 删除序列 ===")
    tree.delete(seq_id=1)
    print(f"删除seq_id=1后: {tree.get_all_sequences()}")
    print(f"统计: {tree.stats}")

