"""
编码题1：文档读取、切块、BGE-M3 稠密+稀疏向量化，并写入 Milvus
运行：python build_knowledge_base.py
"""
import numpy as np
if not hasattr(np.dtypes, "UInt32DType"):
    class FakeUInt32DType:
        name = "uint32"
    np.dtypes.UInt32DType = FakeUInt32DType

if not hasattr(np.dtypes, "UInt64DType"):
    class FakeUInt64DType:
        name = "uint64"
    np.dtypes.UInt64DType = FakeUInt64DType

import hashlib
import os
from datetime import datetime

from pymilvus import MilvusClient, DataType


class DocumentLoader:
    """读取本地 txt 文档"""
    def load_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()


class SimpleEmbedder:
    """本地 BGE-M3 嵌入器：同时生成 dense 与 sparse 向量"""

    def __init__(self, model_path: str = "../models/bge-m3"):
        from FlagEmbedding import BGEM3FlagModel
        self.model_path = model_path
        self.model = BGEM3FlagModel(model_path, use_fp16=False)
        self.dense_dim = 1024

    def _to_milvus_sparse(self, sparse: dict) -> dict[int, float]:
        # TODO 1（3分）：将 BGE-M3 输出的稀疏权重转换为 Milvus 可写入格式
        # 要求：key 转 int，value 转 float，并过滤 value <= 0 的项
        return  {int(k): float(v) for k, v in sparse.items() if float(v) > 0}

    def encode(self, texts: list[str]) -> dict[str, list]:
        """
        返回格式：
        {
            "dense": list[list[float]],
            "sparse": list[dict[int, float]]
        }
        """
        # TODO 2（7分）：调用 BGE-M3 同时生成 dense 与 sparse 向量
        # 提示：
        output = self.model.encode(
              texts,
              batch_size=8,
              max_length=512,
              return_dense=True,
              return_sparse=True,
              return_colbert_vecs=False,
          )
        dense = output["dense_vecs"].tolist()
        sparse_weights = output["lexical_weights"]
        sparse_vecs = [self._to_milvus_sparse(sp) for sp in sparse_weights]
        return {"dense": dense, "sparse": sparse_vecs}


class SimpleChunker:
    """
    父块 + 子块切分器
    返回字段：text / parent_id / parent_content / source / child_index
    """

    def split(
            self,
            text: str,
            source: str = "ai",
            parent_size: int = 200,
            child_size: int = 80,
            overlap: int = 20,
    ) -> list[dict]:
        chunks = []
        # TODO 3（10分）：实现父块+子块切分
        # 要求：
        #   1. overlap 必须小于 child_size，否则抛出 ValueError
        if overlap >= child_size:
            raise ValueError("重叠要小于子块大小！")
        #   3. 每个父块内 child_index 从 0 开始
        parent_index = 0
        for i in range(0, len(text), parent_size):
            parent_content = text[i:i + parent_size]
        #   2. parent_id 格式为 parent_0、parent_1 ...
            parent_id = f"parent_{parent_index}"
            parent_index += 1

            child_index = 0
            step = child_size - overlap  # 每次滑动的步长
            for j in range(0, len(parent_content), step):
                child_text = parent_content[j:j + child_size]
        #   4. 跳过纯空白子块
        #   5. 每个子块必须保留 parent_content
                if not child_text.strip():  # 跳过纯空白
                    continue
                chunk = {
                    "text": child_text,
                    "parent_id": parent_id,
                    "parent_content": parent_content,
                    "source": source,
                    "child_index": child_index,
                }
                chunks.append(chunk)
                child_index += 1
        return chunks


class MilvusIndexer:
    """Milvus 入库器：构建支持混合检索的 Collection"""

    def __init__(
            self,
            uri: str = "http://localhost:19530",
            collection_name: str = "edu_rag_exam_hybrid",
            embedder: SimpleEmbedder | None = None,
    ):
        self.uri = uri
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.embedder = embedder or SimpleEmbedder()

    def create_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            print(f"集合 {self.collection_name} 已存在，已删除旧集合")

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=2048)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.embedder.dense_dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("parent_content", DataType.VARCHAR, max_length=4096)
        schema.add_field("source", DataType.VARCHAR, max_length=32)
        schema.add_field("child_index", DataType.INT64)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=256)
        schema.add_field("created_at", DataType.VARCHAR, max_length=32)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"集合 {self.collection_name} 创建成功")

    def add_chunks(self, chunks: list[dict]):
        """将子块列表写入 Milvus"""
        if not chunks:
            print("无数据可写入")
            return

        texts = [c["text"] for c in chunks]

        # TODO 4（10分）：生成 dense/sparse 向量，构造 data，调用 upsert 写入 Milvus
        # 要求：
        #   1. 调用 self.embedder.encode(texts)
        #   2. dense_vector 写入 dense 结果
        #   3. sparse_vector 写入 sparse 结果
        #   4. id 使用 parent_id + child_index + text 生成 md5
        #   5. embedding_model 写入 self.embedder.model_path
        #   6. created_at 写入当前时间字符串
        # 批量生成向量
        vectors = self.embedder.encode(texts)
        dense_vecs = vectors["dense"]
        sparse_vecs = vectors["sparse"]

        data = []
        for i, chunk in enumerate(chunks):
            unique_str = f"{chunk['parent_id']}_{chunk['child_index']}_{chunk['text']}"
            id_md5 = hashlib.md5(unique_str.encode("utf-8")).hexdigest()
            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = {
                "id": id_md5,
                "text": chunk["text"],
                "dense_vector": dense_vecs[i],
                "sparse_vector": sparse_vecs[i],
                "parent_id": chunk["parent_id"],
                "parent_content": chunk["parent_content"],
                "source": chunk["source"],
                "child_index": chunk["child_index"],
                "embedding_model": self.embedder.model_path,
                "created_at": now_time,
            }
            data.append(record)

        self.client.upsert(collection_name=self.collection_name, data=data)
        print(f"成功写入 {len(chunks)} 条子块")


def main():
    file_path = "../data/knowledge.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"请创建测试文档：{file_path}")

    loader = DocumentLoader()
    raw_text = loader.load_txt(file_path)

    chunker = SimpleChunker()
    chunks = chunker.split(raw_text, source="ai")
    print(f"切分完成，共 {len(chunks)} 个子块")

    indexer = MilvusIndexer(uri="http://192.168.80.1:19530")
    print(f"连接 Milvus：{indexer.uri}")
    indexer.create_collection()
    indexer.add_chunks(chunks)


if __name__ == "__main__":
    main()