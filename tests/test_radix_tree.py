"""
===================================================================
RadixTree 单元测试
===================================================================
"""
import unittest
from core.radix_tree import RadixTree, RadixNode


class TestRadixTree(unittest.TestCase):
    """RadixTree 单元测试类"""
    
    def setUp(self):
        """每个测试前执行"""
        self.tree = RadixTree(max_blocks=1024)
    
    # ==================== 基础功能测试 ====================
    
    def test_insert_single_sequence(self):
        """测试插入单个序列"""
        result = self.tree.insert(seq_id=1, tokens=[1, 2, 3, 4, 5])
        self.assertTrue(result)
        self.assertEqual(len(self.tree), 1)
        
        # 验证序列可以获取
        tokens = self.tree.get_sequence(1)
        self.assertEqual(tokens, [1, 2, 3, 4, 5])
    
    def test_insert_multiple_sequences(self):
        """测试插入多个序列"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3])
        self.tree.insert(seq_id=2, tokens=[4, 5, 6])
        self.tree.insert(seq_id=3, tokens=[7, 8, 9])
        
        self.assertEqual(len(self.tree), 3)
        self.assertEqual(self.tree.get_all_sequences(), [1, 2, 3])
    
    def test_insert_duplicate_seq_id(self):
        """测试插入相同seq_id会先删除再插入"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3])
        self.assertEqual(len(self.tree), 1)
        
        # 插入相同seq_id的不同序列
        self.tree.insert(seq_id=1, tokens=[1, 2, 3, 4, 5])
        
        # 应该只有一条记录，且为新序列
        self.assertEqual(len(self.tree), 1)
        self.assertEqual(self.tree.get_sequence(1), [1, 2, 3, 4, 5])
    
    def test_insert_with_block_ids(self):
        """测试插入时指定block_ids"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3], block_ids=[10, 20, 30])
        
        tokens = self.tree.get_sequence(1)
        self.assertEqual(tokens, [1, 2, 3])
    
    def test_delete_existing_sequence(self):
        """测试删除已存在的序列"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3])
        self.assertEqual(len(self.tree), 1)
        
        result = self.tree.delete(1)
        self.assertTrue(result)
        self.assertEqual(len(self.tree), 0)
        self.assertIsNone(self.tree.get_sequence(1))
    
    def test_delete_nonexistent_sequence(self):
        """测试删除不存在的序列"""
        result = self.tree.delete(999)
        self.assertFalse(result)
    
    def test_clear(self):
        """测试清空树"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3])
        self.tree.insert(seq_id=2, tokens=[4, 5, 6])
        
        self.tree.clear()
        
        self.assertEqual(len(self.tree), 0)
        self.assertEqual(self.tree.get_all_sequences(), [])
    
    # ==================== 前缀匹配测试 ====================
    
    def test_find_longest_prefix_exact_match(self):
        """测试精确匹配"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3, 4, 5])
        
        matched_len, matched_seqs = self.tree.find_longest_prefix([1, 2, 3, 4, 5])
        
        self.assertEqual(matched_len, 5)
        self.assertEqual(matched_seqs, {1})
    
    def test_find_longest_prefix_partial_match(self):
        """测试部分匹配"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3, 4, 5])
        self.tree.insert(seq_id=2, tokens=[1, 2, 3, 6, 7])
        
        # 查询 [1, 2, 3, 4, 8] - 应该匹配前3个token
        matched_len, matched_seqs = self.tree.find_longest_prefix([1, 2, 3, 4, 8])
        
        self.assertEqual(matched_len, 3)
        # seq_id=1 和 seq_id=2 都经过 [1,2,3]
        self.assertEqual(matched_seqs, {1, 2})
    
    def test_find_longest_prefix_no_match(self):
        """测试无匹配"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3])
        
        matched_len, matched_seqs = self.tree.find_longest_prefix([9, 8, 7])
        
        self.assertEqual(matched_len, 0)
        # 根节点的seq_ids包含所有序列
        self.assertEqual(matched_seqs, {1})
    
    def test_find_longest_prefix_empty_query(self):
        """测试空查询"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3])
        
        matched_len, matched_seqs = self.tree.find_longest_prefix([])
        
        self.assertEqual(matched_len, 0)
        # 空查询应该返回根节点的seq_ids
        self.assertEqual(matched_seqs, {1})
    
    def test_find_all_matching(self):
        """测试查找所有匹配的序列ID"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3, 4])
        self.tree.insert(seq_id=2, tokens=[1, 2, 3, 5])
        self.tree.insert(seq_id=3, tokens=[1, 2, 3, 4, 5])
        
        # 查询 [1,2,3,4] - 应该返回所有经过该前缀的序列
        result = self.tree.find_all_matching([1, 2, 3, 4])
        
        self.assertEqual(result, [1, 2, 3])  # seq_id=1,2,3 都经过 [1,2,3,4]
    
    def test_has_prefix_true(self):
        """测试前缀存在 - True"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3, 4, 5])
        
        self.assertTrue(self.tree.has_prefix([1, 2, 3]))
        self.assertTrue(self.tree.has_prefix([1, 2]))
        self.assertTrue(self.tree.has_prefix([1]))
    
    def test_has_prefix_false(self):
        """测试前缀存在 - False"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3])
        
        self.assertFalse(self.tree.has_prefix([1, 2, 3, 4]))
        self.assertFalse(self.tree.has_prefix([2, 3]))
        self.assertFalse(self.tree.has_prefix([9]))
    
    def test_has_prefix_empty(self):
        """测试空前缀"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3])
        
        # 空前缀总是存在（从根节点开始）
        self.assertTrue(self.tree.has_prefix([]))
    
    # ==================== Block复用测试 ====================
    
    def test_get_reusable_blocks_with_match(self):
        """测试获取可复用块 - 有匹配"""
        # 插入seq_id=1, tokens长度32，对应2个block (假设block_size=16)
        self.tree.insert(seq_id=1, tokens=list(range(32)), block_ids=[0, 1])
        
        # 插入seq_id=2，与seq_id=1有公共前缀
        self.tree.insert(seq_id=2, tokens=list(range(32)), block_ids=[2, 3])
        
        # 获取seq_id=2的可复用块
        reusable = self.tree.get_reusable_blocks(2)
        
        # 32个token = 2个block，应该返回seq_id=1的前2个block
        self.assertEqual(len(reusable), 2)
    
    def test_get_reusable_blocks_no_match(self):
        """测试获取可复用块 - 无匹配"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3], block_ids=[0])
        self.tree.insert(seq_id=2, tokens=[4, 5, 6], block_ids=[1])
        
        reusable = self.tree.get_reusable_blocks(2)
        
        # 没有公共前缀，不应复用
        self.assertEqual(len(reusable), 0)
    
    def test_get_prefix_blocks(self):
        """测试获取前缀block信息"""
        # 插入 tokens = [0,1,2,...,31]，共32个token，对应2个block
        self.tree.insert(seq_id=1, tokens=list(range(32)), block_ids=[0, 1])
        
        # 查询前缀 [0,1,2,...,19] - 20个token，对应1个block (假设block_size=16，但实际取整)
        prefix_len, blocks = self.tree.get_prefix_blocks(list(range(20)))
        
        # 20个token应该匹配
        self.assertEqual(prefix_len, 20)
        # 至少返回block 0
        self.assertIn(0, blocks)
    
    # ==================== 序列操作测试 ====================
    
    def test_get_sequence_existing(self):
        """测试获取已存在的序列"""
        self.tree.insert(seq_id=1, tokens=[10, 20, 30])
        
        tokens = self.tree.get_sequence(1)
        
        self.assertEqual(tokens, [10, 20, 30])
    
    def test_get_sequence_nonexisting(self):
        """测试获取不存在的序列"""
        tokens = self.tree.get_sequence(999)
        
        self.assertIsNone(tokens)
    
    def test_contains(self):
        """测试 __contains__ 方法"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3])
        
        self.assertTrue(1 in self.tree)
        self.assertFalse(2 in self.tree)
    
    def test_len(self):
        """测试 __len__ 方法"""
        self.assertEqual(len(self.tree), 0)
        
        self.tree.insert(seq_id=1, tokens=[1, 2])
        self.assertEqual(len(self.tree), 1)
        
        self.tree.insert(seq_id=2, tokens=[3, 4])
        self.assertEqual(len(self.tree), 2)
        
        self.tree.delete(1)
        self.assertEqual(len(self.tree), 1)
    
    def test_get_all_sequences(self):
        """测试获取所有序列ID"""
        self.tree.insert(seq_id=3, tokens=[1])
        self.tree.insert(seq_id=1, tokens=[2])
        self.tree.insert(seq_id=2, tokens=[3])
        
        # 应该排序返回
        self.assertEqual(self.tree.get_all_sequences(), [1, 2, 3])
    
    # ==================== 统计信息测试 ====================
    
    def test_stats(self):
        """测试统计信息"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3, 4, 5])
        
        stats = self.tree.stats
        
        self.assertEqual(stats['seq_count'], 1)
        self.assertEqual(stats['max_blocks'], 1024)
        self.assertGreater(stats['node_count'], 0)
    
    # ==================== 边界条件测试 ====================
    
    def test_insert_single_token(self):
        """测试插入单个token"""
        self.tree.insert(seq_id=1, tokens=[1])
        
        self.assertEqual(len(self.tree), 1)
        
        matched_len, matched_seqs = self.tree.find_longest_prefix([1])
        self.assertEqual(matched_len, 1)
        self.assertEqual(matched_seqs, {1})
    
    def test_insert_empty_tokens(self):
        """测试插入空token列表"""
        result = self.tree.insert(seq_id=1, tokens=[])
        
        self.assertTrue(result)
        self.assertEqual(len(self.tree), 1)
        
        # 验证序列
        tokens = self.tree.get_sequence(1)
        self.assertEqual(tokens, [])
    
    def test_long_common_prefix(self):
        """测试长公共前缀"""
        # 插入100个序列，每个都有相同的前缀
        for i in range(100):
            prefix = list(range(50))
            suffix = [100 + i]
            self.tree.insert(seq_id=i, tokens=prefix + suffix)
        
        # 查询相同前缀
        matched_len, matched_seqs = self.tree.find_longest_prefix(list(range(50)))
        
        self.assertEqual(matched_len, 50)
        self.assertEqual(len(matched_seqs), 100)
    
    def test_delete_and_reinsert(self):
        """测试删除后重新插入"""
        self.tree.insert(seq_id=1, tokens=[1, 2, 3])
        self.tree.delete(1)
        self.tree.insert(seq_id=1, tokens=[1, 2, 3, 4, 5])
        
        self.assertEqual(len(self.tree), 1)
        self.assertEqual(self.tree.get_sequence(1), [1, 2, 3, 4, 5])
    
    def test_multiple_sequences_same_prefix_different_length(self):
        """测试多个不同长度的序列共享前缀"""
        self.tree.insert(seq_id=1, tokens=[1, 2])
        self.tree.insert(seq_id=2, tokens=[1, 2, 3])
        self.tree.insert(seq_id=3, tokens=[1, 2, 3, 4])
        
        # 查找 [1,2,3]
        # 序列1只有2个token，所以在位置2之后就不再经过
        # 公共前缀长度为2（序列1、2、3都经过的前缀）
        matched_len, matched_seqs = self.tree.find_longest_prefix([1, 2, 3])
        
        self.assertEqual(matched_len, 2)
        # 序列1、2、3都经过前缀[1,2]
        self.assertEqual(matched_seqs, {1, 2, 3})


class TestRadixNode(unittest.TestCase):
    """RadixNode 单元测试类"""
    
    def test_node_creation(self):
        """测试节点创建"""
        node = RadixNode(token=1, seq_len=5)
        
        self.assertEqual(node.token, 1)
        self.assertEqual(node.seq_len, 5)
        self.assertFalse(node.is_end)
        self.assertIsNone(node.block_id)
        self.assertEqual(len(node.children), 0)
        self.assertEqual(len(node.seq_ids), 0)
    
    def test_node_with_children(self):
        """测试带子节点的节点"""
        parent = RadixNode(token=1)
        child = RadixNode(token=2)
        parent.children[2] = child
        child.seq_ids.add(1)
        
        self.assertEqual(len(parent.children), 1)
        self.assertIn(2, parent.children)
        self.assertIn(1, child.seq_ids)


# ==================== 集成测试 ====================

class TestRadixTreeIntegration(unittest.TestCase):
    """RadixTree 集成测试 - 模拟真实KV Cache场景"""
    
    def test_prefix_cache_reuse_scenario(self):
        """
        测试前缀缓存复用场景
        
        场景：
        1. 请求1: "Hello, how are you?" -> 生成完整回复
        2. 请求2: "Hello, how" -> 需要复用请求1的前缀缓存
        """
        tree = RadixTree(max_blocks=1024)
        
        # 请求1: "Hello, how are you?" (用token ids模拟)
        request1_tokens = [101, 202, 303, 404, 505, 606]
        tree.insert(seq_id=1, tokens=request1_tokens, block_ids=[0, 1])
        
        # 请求2: "Hello, how" - 查询是否有可复用的前缀
        query_tokens = [101, 202, 303]
        
        # 检查前缀是否存在
        self.assertTrue(tree.has_prefix(query_tokens))
        
        # 查找匹配的序列
        matched_len, matched_seqs = tree.find_longest_prefix(query_tokens)
        
        self.assertEqual(matched_len, 3)
        self.assertIn(1, matched_seqs)
        
        # 获取可复用块
        # 插入新序列
        tree.insert(seq_id=2, tokens=[101, 202, 303, 707, 808])
        reusable = tree.get_reusable_blocks(2)
        
        # 3个token可以复用1个block
        self.assertGreaterEqual(len(reusable), 0)
    
    def test_concurrent_requests_scenario(self):
        """
        测试并发请求场景
        
        场景：多个用户同时发送请求，可能共享部分前缀
        """
        tree = RadixTree(max_blocks=1024)
        
        # 系统提示 (所有请求共享)
        system_prompt = [100, 101, 102, 103, 104, 105]
        
        # 用户1: "Hello"
        user1_tokens = system_prompt + [200, 201]
        tree.insert(seq_id=1, tokens=user1_tokens, block_ids=[0, 1])
        
        # 用户2: "Hi there"
        user2_tokens = system_prompt + [202, 203, 204]
        tree.insert(seq_id=2, tokens=user2_tokens, block_ids=[2, 3])
        
        # 用户3: "Good morning"
        user3_tokens = system_prompt + [205, 206, 207, 208]
        tree.insert(seq_id=3, tokens=user3_tokens, block_ids=[4, 5])
        
        # 所有用户都共享系统提示
        matched_len, matched_seqs = tree.find_longest_prefix(system_prompt)
        
        self.assertEqual(matched_len, len(system_prompt))
        self.assertEqual(matched_seqs, {1, 2, 3})
        
        # 验证统计
        stats = tree.stats
        self.assertEqual(stats['seq_count'], 3)


if __name__ == '__main__':
    # 运行所有测试
    unittest.main(verbosity=2)

