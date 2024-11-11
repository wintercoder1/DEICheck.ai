# DEICheck.ai
The retrieval augmented generative AI powered app that fuels DEICheck.ai. Uses Llama with Llama Index and Pinecone Vector Database. (Coming soon... for now it uses LLamaIndex default in memory vector store for word embeddings instead.)

## What it does

This app will answer one of a couple questions: what is the lean of a public figure or brand? What does a public figure or brand think of DEI (and why do they consider that a good thing)? Are they woke? We will use AI to give our users the answer to these questions. The AI will summarized sources and tell us which ones are relevant to this query. 

(We are pro DEI btw..)

#### With Citation (why this imporves it)

Instead of only giving one answer we can give the answer along with links to new articles or financial contribution data. This is comparable to the difference between asking ChatGPT and Perplexity. Perplexity will give and answer followed by a list of sources that the answer came from. In my opinion this is a way more compeling way to give an answer. It creates trust instead of telling the user to beleive the magic number with the assurance 'trust me bro its AI' The answer we give will make sense and be verifiable.

#### Without Citation

Making it work with the citations above is more challenging to implement and depends heavily on data quality. It is also more comutationaly expensive. Therefore as a fallback this will be able to answer without the citation. It is easier for the LLM to get this correct though it is objectively the worse way to answer.

## Instructions

Bring your own LLM with weights saved in .gguf format. Place that in a folder in the main directory called weights.

One recomendation is the Llama Instruct fine tuned model. It can be found here: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct. 

### MacOS/Linux
Begin the FastAPI server with the command:

fastapi dev API.py

### GPU acceleration on Windows

In addition to the dependencies in requirements.txt we will need to install Torch with CUDA enabled:

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

(official doc: https://pytorch.org/get-started/locally/)

Set following environment variables:

CMAKE_ARGS = "-DLLAMA_OPENBLAS=on"
FORCE_CMAKE = 1

pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

Once that is complete the politicalQueryWithOUTCitationLocalWithGPU method will work with the cpu. The response will be 20-30 seconds instead of minutes.

To start the FastAPI server to work on Windows the follow command will need to be run:

uvicorn API:app --reload

Note: This is a different command than on MacOS/Linux

## Once setup is Complete
This will launch the server with the endpoint: getPoliticalLeaningWithCitation/{query_topic}

Replace query topic with whatever you want to know more about polically for example a get request on
{base_url}/getPoliticalLeaningWithCitation/Valvline will tell you about what Valvoline's political leaning is and who they have donated to.
