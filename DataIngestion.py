import os
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser

def crawl_data_folder_load_to_index(folder_path: str):
        test_limit = 20000
        test_i = 0
        subdirs = [x[0] for x in os.walk(folder_path)]
        folder_names = []
        index_list = []

        for dir in subdirs:

            if dir == './data':
                 continue

            if test_i >= test_limit:
                break
            test_i += 1
            
            split = dir.split('/')
            topic_name_folder  = split[ -1]
            print(topic_name_folder)
            try:
                documents = SimpleDirectoryReader(dir).load_data()
                index = GPTVectorStoreIndex.from_documents(
                    documents,
                )
                index_list.append(index)
                folder_names.append(topic_name_folder)
            except ValueError:
                print('No files found in: ' + dir)
                continue
        return index_list, folder_names

def crawl_data_folder_get_documents(folder_path: str):
        test_limit = 20000
        test_i = 0
        subdirs = [x[0] for x in os.walk(folder_path)]
        folder_names = []
        docuemnt_list = []

        for dir in subdirs:

            if dir == './data':
                 continue

            if test_i >= test_limit:
                break
            test_i += 1
            
            split = dir.split('/')
            topic_name_folder = split[ -1]
            print(topic_name_folder)
            try:
                documents = SimpleDirectoryReader(dir).load_data()
                docuemnt_list.append(documents)
                folder_names.append(topic_name_folder)
            except ValueError:
                print('No files found in: ' + dir)
                continue
        return docuemnt_list, folder_names


def createNewsDocumentsIndex(reCreateIndex=False):
    if not os.path.exists("./citation") or reCreateIndex == True:
        document_list, topics = crawl_data_folder_get_documents("./data")

        index = VectorStoreIndex.from_documents(
            document_list[0],
        )
        for doc in document_list[1:]:
            parser = SimpleNodeParser()
            new_nodes = parser.get_nodes_from_documents(doc)
            index.insert_nodes(new_nodes)
        index.storage_context.persist(persist_dir="./citation")
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./citation"),
        )
    return index