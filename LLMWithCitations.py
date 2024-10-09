import DataIngestion
# import prompt_templates
from PromptTemplates import POLITICAL_LIB_OR_CON_SCORE_PROMPT, PROMPT_ONLY_POLICAL_LEANING, SYSTEM_ONLY_POLITICAL_LIB_OR_CON_SCORE_PROMPT

from llama_index.core import PromptTemplate
from llama_index.core import Settings
# from llama_index import GPTListIndex
from llama_index.core.indices.list import GPTListIndex
from llama_index.core.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.openai import OpenAI

#
#
# This is a RAG that uses the LLama Index in memory vector store to create a query engine.
#
#
# TODO: Check for documents in index first before querying the query engine.
# TODO: Fall back to non query engine wrapped LLM on case were no relevant documents are found for topic.
# TODO: Replace default vecotr store with Pinecone.
# TODO: Replace default gpt-3.5 with open source Llama model.
# TODO: Create new index for financial data. Use Reciprocal Rerank Fusion Retriever¶:
#       https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion/#reciprocal-rerank-fusion-retriever
# TODO: Find more test documents. (Later populate these with a web crawler) 
#
class LLMWithCitations():

    def __init__(self) -> None:
        # The LLM version
        Settings.llm = self.confifureLlamaCPP()
        # The embeddings model
        bge_small_embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')
        Settings.embed_model = bge_small_embed_model

        # self.news_document_index = DataIngestion.createNewsDocumentsIndex(reCreateIndex=True)
        self.news_document_index = DataIngestion.createNewsDocumentsIndex()
        
        self.citation_query_engine = CitationQueryEngine.from_args(
            self.news_document_index,
            similarity_top_k=5,
            # here we can control how granular citation sources are, the default is 512
            citation_chunk_size=128,
        )
        return

    def confifureLlamaCPP(self):
        # llama_3_8B_instruct_base_path = '/Users/steve/Documents/Dev/Llama/Meta-Llama-3-8B-Instruct-GGUF/'
        llama_3_8B_instruct_base_path = 'weights/'
        # llama_3_8B_instruct_path = llama_3_8B_instruct_base_path + 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf'
        # llama_3_8B_instruct_path = llama_3_8B_instruct_base_path + 'Meta-Llama-3-8B-Instruct-Q4_K_S.gguf'
        llama_3_8B_instruct_path = llama_3_8B_instruct_base_path + 'Meta-Llama-3-8B-Instruct-Q3_K_L.gguf'
        # llama_3_8B_instruct_path = llama_3_8B_instruct_base_path + 'Meta-Llama-3-8B-Instruct-Q3_K_M.gguf'
        # llama_3_8B_instruct_path = llama_3_8B_instruct_base_path + 'Meta-Llama-3-8B-Instruct-IQ4_NL.gguf'
        
        llm = LlamaCPP(
            # You can pass in the URL to a GGML model to download it automatically
            # model_url=model_url,
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            # model_path=llama_2_path,
            model_path=llama_3_8B_instruct_path,
            # temperature=1,
            max_new_tokens=128,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            # context_window=3900,
            # kwargs to pass to __call__()
            generate_kwargs={},
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            # model_kwargs={"n_gpu_layers": 1},
            # transform inputs into Llama2 format
            # messages_to_prompt=messages_to_prompt,
            # completion_to_prompt=completion_to_prompt,
            # verbose=True,
            verbose=False,
        )
        return llm

    #
    # Takes query topic and perfomrs inference with political leaning prompt.
    # Gives ciations with query engine. 
    # Not 100% accurate yet. Use politicalQueryWithOUTCitation if thats an issue.
    def politicalQueryWithCitation(self, topic):
        prompt_tmpl = PromptTemplate(POLITICAL_LIB_OR_CON_SCORE_PROMPT)
        the_query = prompt_tmpl.format(topic_of_prompt=topic)

        response = self.citation_query_engine.query(the_query)
        responseStr = str(response)

        if (self.indexedInfoNotConnectedToTopic(responseStr)):
            response = self.politicalQueryWithOUTCiation(topic)
            citation_string = '\nCitations: None'
            responseStr = str(response) + citation_string
        else:
            # source nodes are 6, because the original chunks of 1024-sized nodes were broken into more granular nodes
            citation_string = '\nCitations:\n' + str(response.source_nodes[0].node.get_text())
            responseStr += citation_string

        return responseStr

    def politicalQueryWithOUTCitation(self, topic):
        prompt_tmpl = PromptTemplate(POLITICAL_LIB_OR_CON_SCORE_PROMPT)
        the_query = prompt_tmpl.format(topic_of_prompt=topic)

        response = Settings.llm.complete(the_query)

        return response
    
    # Test wether the response string is telling us that the query engine could not fetch relevant documents.
    # Note that the query engine halucinates often and returns many false negatives.
    # A return type of True means revelant documents or sources were NOT found.
    def indexedInfoNotConnectedToTopic(self, responseStr):
        match = (
            responseStr.startswith('There is no relevant information in the provided sources ') or
            responseStr.startswith('I\'m sorry, none of the provided sources contain information about') or 
            responseStr.startswith('I\'m sorry, but the provided sources') or 
            responseStr.startswith('I\'m sorry, but none of the provided sources') or
            responseStr.startswith('Unfortunately, none of the provided sources') or 
            responseStr.startswith('Unfortunately, the provided sources do not') or
            responseStr.startswith('Sorry, none of the provided sources') or 
            responseStr.startswith('I am unable to provide an answer based on the provided sources') or
            responseStr.startswith('There is no information')
        )
        return match


def testWithTopic():
    llmWithCitations = LLMWithCitations()

    # Test topics
    # topic = "Barnes and Noble"
    # topic = "Black Rifle Coffee"
    # topic = "BP"
    # topic = "Bud Light"
    # topic = 'Diddy'
    topic = 'Jiffy Lube'
    # topic = 'Molson'
    # topic = 'Valvoline'
    response = llmWithCitations.politicalQueryWithCitation(topic)

    return response

if __name__ == "__main__":
    resp = testWithTopic()
    print('\n' + resp + '\n')