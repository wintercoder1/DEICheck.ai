from fastapi import FastAPI
from LLMWithCitations import LLMWithCitations

app = FastAPI()


# @app.get("/")
# async def root():
    # html = "
    #     <form action="" method="get" class="form-example">
    #         <div class="form-example">
    #             <label for="name">Enter your name: </label>
    #             <input type="text" name="name" id="name" required />
    #         </div>
    #     </form>
    # "
    # return {"message": "Hello World"}



@app.get("/getPoliticalLeaningWithCitation/{query_topic}")
async def rgetPoliticalLeaningWithCitation(query_topic):
    llmWithCitations = LLMWithCitations()
    query_topic_str = str(query_topic)
    reposne = llmWithCitations.politicalQueryWithCitation(query_topic_str)
    return {"Response": reposne}


@app.get("/getPoliticalLeaningWithoutCitation{query_topic}")
async def getPoliticalLeaningWithoutCitation(query_topic):
    llmMWithCitations = LLMWithCitations()
    query_topic_str = str(query_topic)
    reposne = llmMWithCitations.politicalQueryWithOUTCitation(query_topic_str)
    return {"Response": reposne}