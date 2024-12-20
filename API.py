from fastapi import FastAPI
import uvicorn
from DataCache.CassandraDBCache import CassandraDBCache
from DataClassWrappers.TopicInfo import TopicInfo
from LLMQueryEngine import LLMQueryEngine
import Util


app = FastAPI()

isProd = True

@app.get("/getPoliticalLeaningWithCitation/{query_topic}")
async def getPoliticalLeaningWithCitation(query_topic, overrideCache: bool | None = None):
    # return {"message": "Hello, FastAPI!"}
    if overrideCache:
        override = overrideCache
    else:
        override = False
    json = withCitation(query_topic, overrideCache=override)
    return json

# Will not return a citation based off of sources. Won't consider the documents gathered on possible topics.
@app.get("/getPoliticalLeaning/{query_topic}")
async def getPoliticalLeaningWithoutCitation(query_topic):
    # If this was already answered return the cached response and return
    dbCache = CassandraDBCache(prod=isProd)
    most_recent = dbCache.fetchInfoOnTopicMostRecent(query_topic)
    if most_recent != None: # cached answer found.
        print('returning cached response: ')
        json = Util.escapedJsonFromTopicInfo(most_recent, cached=True)
        print(json)
        return json

    llmQueryEngine = LLMQueryEngine()
    query_topic_str = str(query_topic)
    response = llmQueryEngine.politicalQueryWithOUTCiation(query_topic_str)
    response_dataclass = Util.parsePolitcalLeaingResponse(response, query_topic, citation=False)

    # Save to DB if a properly formatted answer.
    if type(response_dataclass) is TopicInfo: 
        dbCache.writeTopicInfoToDB(response_dataclass)
    else:# If the answer cannot be parsed give back the original string.
        return {'response': response_dataclass}
    
    json = Util.escapedJsonFromTopicInfo(most_recent, cached=False)
    print()
    print(json)
    return json

# Gpu enabled version. Local only. Currently does not fetch citations from index
# Note we wont cache local responses as they won't cost us as much to w.e.
@app.get("/getPoliticalLeaningWithGPU/{query_topic}")
async def getPoliticalLeaningWithoutCitationWithGPU(query_topic):
    llmQueryEngine = LLMQueryEngine()
    query_topic_str = str(query_topic)
    response = llmQueryEngine.politicalQueryWithGPULocal(query_topic_str)
    response_dataclass = Util.parsePolitcalLeaingResponse(response, query_topic)
    
     # Save to DB if a properly formatted answer.
    if response_dataclass is not TopicInfo:
        return {'response': response_dataclass}
     
    json = Util.escapedJsonFromTopicInfo(response_dataclass)
    print(json)
    return json


# Code for the ciation response
# TODO: Move to its ownfile. This file should be an api only wrapper around the main logic.
def withCitation(query_topic: str, overrideCache: bool = False):
    dbCache = CassandraDBCache(prod=isProd)
    # If not opted out of Cache response
    if not overrideCache:
        # If this was already answered return the cached response and return
        most_recent = dbCache.fetchInfoOnTopicMostRecent(query_topic)
        if most_recent != None: # cached answer found.
            print('returning cached response: ')
            json = Util.escapedJsonFromTopicInfo(most_recent, cached=True)
            print(json)
            return json

    # Otherwise get the LLM response.
    llmWithCitations = LLMQueryEngine()
    query_topic_str = str(query_topic)
    print('Request received with topic: {query_topic}')
    print('Waiting....')
    response = llmWithCitations.politicalQueryWithCitation(query_topic_str)
    # Format the reponse and parse out the important information.
    response_dataclass:TopicInfo = Util.parsePolitcalLeaingResponse(response, query_topic)
 
    # Save to DB if a properly formatted answer.
    if type(response_dataclass) is TopicInfo: # will be str or null if parse error.
        print('Writing to DB now..')
        dbCache.writeTopicInfoToDB(response_dataclass)
    else: # If the answer cannot be parsed give back the original string.
        return {'response': response_dataclass}
    # Convert to json to give back to client.
    
    json = Util.escapedJsonFromTopicInfo(response_dataclass, cached=False)
    print()
    print(json)
    return json

# Use this ewith the uvicorn web server.
if __name__ == "__main__":
    uvicorn.run('API:app', host="127.0.0.1", port=8000, reload=True)