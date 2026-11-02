#该文件用于帮助rag_qa这个Python包，管理包内的脚本，方便其它包的调用。
import os, sys

#1.记录当前目录.
current_dir = os.path.abspath(__file__)
# print(f'current_dir: {current_dir}')
#2.记录rag_qa目录->Milvus版的问答系统的目录.
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path: {rag_qa_path}')
#3.添加 rag_qa_path路径到 系统路径变量中.
sys.path.insert(  0, rag_qa_path)

#4.具体的导包动作
from core.prompts import RAGPrompts
from core.vector_store import VectorStore
from core.rag_pipeline import RAGSystem     # RAG系统的核心代码 ->添加历史记录 和 流式输出时选择.
