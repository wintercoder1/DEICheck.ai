import json
from fastapi import FastAPI
from LLMWithCitations import LLMWithCitations

app = FastAPI()

@app.get("/getPoliticalLeaningWithCitation/{query_topic}")
async def getPoliticalLeaningWithCitation(query_topic):

    llmWithCitations = LLMWithCitations()
    query_topic_str = str(query_topic)
    print('Request received with topic: {query_topic}')
    print('Waiting....')

    reposne = llmWithCitations.politicalQueryWithCitationLocal(query_topic_str)
    split = reposne.split('Lean:')

    if len(split) > 1:
        split = split[1]
    else:
        split = split[0]
    split = split.split('Rating:')
    leanStr = split[0]

    split = split[1]
    split = split.split('Context:')
    ratingStr = split[0]

    split = split[1]
    split = split.split('Citations:')
    contextStr = split[0]

    if len(split) > 0:
        citationStr = split[1]
    else :
        citationStr = 'Citation: None'

    response = {
        'lean': leanStr,
        'number': ratingStr,
        'context': contextStr,
        'citation': citationStr
    }

    print(response)

    return {"Response": response}


@app.get("/getPoliticalLeaningWithoutCitation/{query_topic}")
async def getPoliticalLeaningWithoutCitation(query_topic):
    llmMWithCitations = LLMWithCitations()
    query_topic_str = str(query_topic)
    reposne = llmMWithCitations.politicalQueryWithOUTCitationLocal(query_topic_str)
    return {"Response": reposne}