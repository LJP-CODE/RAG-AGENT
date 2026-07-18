"""
余弦相似度 (Cosine Similarity) 实现
====================================
计算两个向量之间夹角的余弦值，值域 [-1, 1]，
值越接近 1 表示两个向量越相似。

公式: cos(A, B) = (A · B) / (||A|| * ||B||)
"""

import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def cosine_similarity_vector(vec_a: list[float], vec_b: list[float]) -> float:
    """
    计算两个向量的余弦相似度。

    参数:
        vec_a: 第一个向量
        vec_b: 第二个向量

    返回:
        余弦相似度值（-1 到 1 之间）

    抛出:
        ValueError: 向量长度不同或为零向量
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"向量长度不一致: {len(vec_a)} vs {len(vec_b)}")

    if len(vec_a) == 0:
        raise ValueError("向量不能为空")

    # 计算点积: A · B = Σ(Ai * Bi)
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # 计算各自的模: ||A|| = sqrt(Σ(Ai²))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("零向量无法计算余弦相似度")

    return dot_product / (norm_a * norm_b)


def cosine_similarity_numpy(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    使用 NumPy 计算两个向量的余弦相似度（更高效）。
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return float(dot_product / (norm_a * norm_b))


def text_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    计算两段文本的余弦相似度。

    使用 TF-IDF 将文本转为向量后计算余弦相似度。

    参数:
        text_a: 第一段文本
        text_b: 第二段文本

    返回:
        余弦相似度值（0 到 1 之间，TF-IDF 无负值）
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
    # 将稀疏矩阵转为稠密向量
    vec_a = tfidf_matrix[0].toarray().flatten()
    vec_b = tfidf_matrix[1].toarray().flatten()
    return cosine_similarity_numpy(vec_a, vec_b)


# ==================== 示例演示 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("1. 基础向量余弦相似度")
    print("=" * 50)

    # 示例 1: 完全相同的方向
    v1 = [1, 2, 3]
    v2 = [2, 4, 6]
    sim = cosine_similarity_vector(v1, v2)
    print(f"  v1 = {v1}")
    print(f"  v2 = {v2}")
    print(f"  相似度: {sim:.4f}  (应为 1.0，方向完全相同)")
    print()

    # 示例 2: 垂直向量
    v3 = [1, 0]
    v4 = [0, 1]
    sim = cosine_similarity_vector(v3, v4)
    print(f"  v3 = {v3}")
    print(f"  v4 = {v4}")
    print(f"  相似度: {sim:.4f}  (应为 0.0，互相垂直)")
    print()

    # 示例 3: 相反方向
    v5 = [1, 1]
    v6 = [-1, -1]
    sim = cosine_similarity_vector(v5, v6)
    print(f"  v5 = {v5}")
    print(f"  v6 = {v6}")
    print(f"  相似度: {sim:.4f}  (应为 -1.0，方向完全相反)")
    print()

    # 示例 4: 任意角度
    v7 = [3, 4]
    v8 = [1, 2]
    sim = cosine_similarity_vector(v7, v8)
    print(f"  v7 = {v7}")
    print(f"  v8 = {v8}")
    print(f"  相似度: {sim:.4f}")
    print()

    print("=" * 50)
    print("2. NumPy 实现（结果一致）")
    print("=" * 50)
    sim_np = cosine_similarity_numpy(np.array(v7), np.array(v8))
    print(f"  numpy 相似度: {sim_np:.4f}")
    print()

    print("=" * 50)
    print("3. 文本余弦相似度（TF-IDF）")
    print("=" * 50)

    docs = [
        ("我喜欢编程和写代码", "我热爱编码和写程序"),
        ("今天天气真好", "今天天气真糟糕"),
        ("苹果是一种水果", "MacBook 是一台电脑"),
    ]

    for text_a, text_b in docs:
        sim = text_cosine_similarity(text_a, text_b)
        print(f"  文本 A: {text_a}")
        print(f"  文本 B: {text_b}")
        print(f"  相似度: {sim:.4f}")
        print()
