# 该脚本用于:处理输入文档，分块并准备向量存储.
import numpy as np
if not hasattr(np.dtypes, "UInt32DType"):
    class FakeUInt32DType:
        name = "uint32"
    np.dtypes.UInt32DType = FakeUInt32DType

if not hasattr(np.dtypes, "UInt64DType"):
    class FakeUInt64DType:
        name = "uint64"
    np.dtypes.UInt64DType = FakeUInt64DType

import os
from langchain_community.document_loaders import TextLoader     # 加载纯文本文件(.txt)用的工具
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader        # 加载(.md)文件的工具，保留标题，列表等结构化信息
from langchain_text_splitters import MarkdownTextSplitter       # 针对.md文档的文本分割器.
from datetime import datetime           # 用于记录文档加载/处理时间.
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter        # 适配中文教育文本层级结构
from rag_qa.edu_text_spliter import AliTextSplitter                     # 阿里中文优化(教育场景专用文本分割器)
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader      # 教育场景专用文档加载器(支持OCR,处理扫描版课件，图片文字等.)
from base.config import Config
from base.logger import logger

# Notebook 加载器(.ipynb)；依赖缺失时降级 TextLoader
try:
    from langchain_community.document_loaders import NotebookLoader
except ImportError:
    NotebookLoader = None
    logger.warning("NotebookLoader 不可用，.ipynb 将降级为 TextLoader 加载")

# todo 1.定义配置类 获取配置参数.
# 1.创建配置实例.
conf = Config()
# 2.定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # 代码/配置文本文件使用 TextLoader(utf-8)
    ".py": TextLoader,
    ".json": TextLoader,
    ".yaml": TextLoader,
    ".yml": TextLoader,
    ".toml": TextLoader,
    # Notebook(.ipynb)，依赖缺失时降级 TextLoader
    ".ipynb": NotebookLoader if NotebookLoader is not None else TextLoader,
    # Markdown(MDX) 文档，按原格式以文本加载(不做转换)
    ".mdx": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}

# 文档集元数据配置(仅保存 repo/version/url_base，完整 URL 在 citation 阶段生成)
DOCSET_META = {
    "langchain": {"repo": "langchain-ai/langchain", "version": "main",
                  "url_base": "https://github.com/langchain-ai/langchain", "language": "en"},
    "milvus": {"repo": "milvus-io/milvus-docs", "version": "master",
               "url_base": "https://github.com/milvus-io/milvus-docs", "language": "zh/en"},
    "bge": {"repo": "FlagOpen/FlagEmbedding", "version": "master",
            "url_base": "https://github.com/FlagOpen/FlagEmbedding", "language": "en"},
    "transformers": {"repo": "huggingface/transformers", "version": "main",
                     "url_base": "https://github.com/huggingface/transformers", "language": "en"},
}


def _infer_doc_type(file_path: str) -> str:
    """按优先级推断文档类型：README > 路径关键字(examples/tutorial/api/reference) > 扩展名。"""
    p = file_path.replace("\\", "/").lower()
    name = os.path.basename(p)
    if "readme" in name:
        return "readme"
    if "examples" in p:
        return "example"
    if "tutorial" in p:
        return "tutorial"
    if "api" in p:
        return "api"
    if "reference" in p:
        return "reference"
    ext = os.path.splitext(p)[1]
    if ext == ".py":
        return "code"
    if ext == ".ipynb":
        return "notebook"
    if ext in (".json", ".yaml", ".yml", ".toml"):
        return "config"
    return "markdown"


# todo 2.核心加载函数模块 -> 从指定目录递归加载所有支持类型文件，添加原数据并返回文档列表。
def load_document_from_directory(directory_path):
    """
    从指定目录(含子目录)加载所有支持类型的文件，为每个文档添加原数据
    :param directory_path:目标文件夹的绝对路径.
    :return:加载完成的文档列表,每个元素为 LangChain Document对象,含:page_content 和 metadata
    """
    # 1.初始化空列表 -> 用于存储所有加载成功的文档.
    documents = []
    # 2.获取支持的文件扩展名结合
    supported_extensions = document_loaders.keys()
    # 3.从目录名提取‘学科类别'原数据，例如:'ai_data'->提取'ai'作为source
    source = os.path.basename(directory_path).replace('_data', '')
    # 4.递归遍历目标目录及其所有子目录，注意:os.walk()返回 当前目录路径，子目录列表，文件列表.
    for root, _, files in os.walk(directory_path):
        # 5.遍历当前目录下的每个文件，逐一处理.
        for file in files:
            # 5.1构造文件的绝对路径 root+文件名，避免相对路径问题.
            file_path = os.path.join(root, file)
            # print(f'file path: {file_path}')   # file path: ../data/ai_data\LLM基础知识.pdf

            # 5.2提取文件扩展名并转为小写 ->统一格式.
            file_extension = os.path.splitext(file)[1].lower()
            # print(f'file extension: {file_extension}')      # file extension: .pdf

            # 5.3 若文件类型在支持列表中，执行加载逻辑
            if file_extension in supported_extensions:
                try:
                    # 5.3.1 根据文件扩展名获取对应的加载器 -> 创建加载器实例.
                    loader_class = document_loaders[file_extension]
                    # 5.3.2实例化加载器
                    if file_extension in ('.txt', '.py', '.json', '.yaml', '.yml', '.toml', '.mdx'):
                        loader = loader_class(file_path, encoding='utf-8')  # 文本类强制使用utf-8编码
                    elif file_extension == '.ipynb' and NotebookLoader is None:
                        loader = loader_class(file_path, encoding='utf-8')  # 降级为 TextLoader
                    else:
                        loader = loader_class(file_path)
                    # 5.3.3调用加载器的Load()方法，加载文件内容->返回文档列表，单个文件可能拆分成多个Document.
                    loaded_docs = loader.load()
                    # print(f'加载文件:{loaded_docs}')

                    # 5.3.4给每个加载的文档添加元数据 -> 便于后续RAG检索时过滤/溯源.
                    for doc in loaded_docs:
                        meta = DOCSET_META.get(source, {})
                        rel_path = os.path.relpath(file_path, directory_path).replace("\\", "/")
                        # 文档集类别，例如:langchain
                        doc.metadata['source'] = source
                        # 相对 docset 目录的文件路径(用于 citation)
                        doc.metadata['file_path'] = rel_path
                        # 文档集元数据(缺失时用空串兜底，避免 Milvus VARCHAR 为空)
                        doc.metadata['repo'] = meta.get('repo', '')
                        doc.metadata['version'] = meta.get('version', '')
                        doc.metadata['language'] = meta.get('language', '')
                        doc.metadata['doc_type'] = _infer_doc_type(rel_path)
                        doc.metadata['title'] = ''
                        doc.metadata['section'] = ''
                        # 为文档添加当前时间戳元数据，isoformat()->精确到秒
                        doc.metadata['timestamp'] = datetime.now().isoformat()
                    # 5.3.5 将添加元数据后的文档列表添加到总列表中。
                    documents.extend(loaded_docs)
                    # 5.3.6记录INF0日志
                    logger.info(f'成功加载文件:{file_path}')
                except Exception as e:
                    # 5.3.7捕获异常 ->输出异常信息，继续处理下一个文件.
                    logger.error(f'加载文件 {file_path} 失败:{str(e)}')
            else:
                # 5.4 若文件不再支持列表，记录警告日志.
                logger.warning(f'不支持的文件类型:{file_path}')
    return documents
# todo 3.文档处理核心函数 ->加载文档后进行分层切分(父块+子块):返回带元数据的子块列表
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                     child_chunk_size=conf.CHILD_CHUNK_SIZE,
                     chunk_overlap=conf.CHUNK_OVERLAP):
    """
    加载文档并进行分层切分:先切成大粒度父块，再将父块切成小粒度子块，子块关联父块元数据.
    :param directory_path:文档目录路径
    :param parent_chunk_size:父块切分长度(大粒度,例如:1200)
    :param child_chunk_size:子块切分长度(小粒度,例如:300)
    :param chunk_overlap:子块重叠长度，确保上下文关联，例如:50
    :return:child_chunks:切分后的子块列表(每个子块含父块关联信息)
    """
    # 从指定目录加载所有文档
    documents = load_document_from_directory(directory_path)
    # 记录加载的文档总数日志
    logger.info(f"加载的文档数量: {len(documents)}")

    # 初始化父块和子块分词器（通用）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化 Markdown 专用分词器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 3.初始化子块列表，存储最终切分结果.
    child_chunks = []
    # 4.遍历每个原始文档 ->带索引i，用于生成唯一ID
    for i,doc in enumerate(documents):
        # 4.1 获取文件扩展名 ->用于判断是否是MarkDown文件，选择对应的切分器.
        file_extension = os.path.splitext(doc.metadata.get("file_path",""))[1].lower()
        is_markdown = (file_extension == ".md")     # 标记是否是MarkDown文件
        # 4.2根据文件类型选择切分器
        parent_splitter_to_use=markdown_parent_splitter if is_markdown else parent_splitter     # 父块切分器
        child_splitter_to_use=markdown_child_splitter if is_markdown else child_splitter        # 子块切分器
        logger.info(f'处理文档:{doc.metadata["file_path"]},用切分器:{"Narkdown" if is_markdown else "ChineseRecursive(中文递归)-通用"}')
        # 4.3将文档切分成父块 大粒度，保留较多上下文
        # parent_docs的格式:[父块1Document对象，父块2Document对象，...]
        parent_docs=parent_splitter_to_use.split_documents([doc])       # 传入列表格式，split_documents()函数要的是可迭代类型.

        # 5.遍历每个父块 父块切分成子块 ->小粒度，减少上下文关联.
        for j,parent_doc in enumerate(parent_docs):
            # 5.1生成父块唯一id，格式:doc_文档索引_parent_父块索引
            parent_id=f"doc_{i}_parent_{j}"
            parent_doc.metadata["parent_id"]=parent_id
            parent_doc.metadata["content"]=parent_doc.page_content      # 存储父块完整内容.
            # 如下输出格式为:父块:../data/ai_data\LLM基础知识.pdf,索引:9,唯一ID:doc_0_parent_9
            # print(f'父块: {parent_doc.metadata["file_path"]}, 索引: {j}, 唯一ID: {parent_id}')
            # 5.3 将父块切分成子块 ->小粒度，并于精准匹配.
            sub_chunks=child_splitter_to_use.split_documents([parent_doc])

            # 6.遍历每个子块 添加父块关联信息.
            for k,sub_chunk in enumerate(sub_chunks):
                # 6.1添加子块关联父块信息->父块ID，父块内容，子块内容.
                sub_chunk.metadata["parent_id"]=parent_id                        # 关联父块ID
                sub_chunk.metadata["parent_content"]=parent_doc.page_content      # 关联父块的完整内容.
                # 6.2生成子块唯一的ID，格式:父块ID_chid_子块索引.
                sub_chunk.metadata["id"] = f'{parent_id}_child_{k}'
                child_chunks.append(sub_chunk)
    # 7.记录日志.
    logger.info(f"切分完成的子块总数:{len(child_chunks)}")
    return child_chunks


if __name__ == '__main__':
    directory_path = '../data/ai_data'
    # documents = load_document_from_directory(directory_path)
    # print(f'加载完成的文档总数:{len(documents)}')
    # print(f'前1个文档详情:{documents[0] if documents else "未加载到任何文档"}')
    # 4.调用文档处理函数 加载指定目录并切分文档.
    chunks=process_documents('../data/ai_data',conf.PARENT_CHUNK_SIZE,conf.CHILD_CHUNK_SIZE,conf.CHUNK_OVERLAP)
    print(f'切分完成的子块总数:{len(chunks)}')
    print(f'前1个子块详情:{chunks[0] if chunks else "未切分到任可子块"}')
