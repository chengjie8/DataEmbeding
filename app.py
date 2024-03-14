import json
import os
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import Pinecone
import uuid

load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "helloworld"
namespace="https://helloworld-m1fp81f.svc.gcp-starter.pinecone.io"
index = pc.Index(index_name)
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")


def describe_index(index_name):
    if index_name not in pc.list_indexes().names():
            # Do something, such as create the index
            pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=PodSpec(
                            environment='us-east1-gcp'
                    )
            )
    return pc.describe_index(index_name)

def embed_data():
    # Load JSON file
    with open('output.json', 'r') as file:
        data = json.load(file)

    # Process each item in the JSON file
    for i, item in enumerate(data):
        html_content = item['html']
        metadata = item
        embedding = embedder.embed_query(html_content)
        print(len(embedding))
        id_str = str(uuid.uuid4())
        index.upsert(
            vectors = [
                (id_str, embedding, metadata)
            ],
            namespace="https://helloworld-m1fp81f.svc.gcp-starter.pinecone.io"
        )

def similarity_search(query):
    embedding = embedder.embed_query(query)
    results = index.query(namespace=namespace, include_metadata=True, top_k=3, vector=embedding, include_values=True)
    best_matched_documents = []

    for match in results.matches:
        doc_id = match["id"]
        best_matched_documents.append(match["metadata"]["html"])

    return best_matched_documents


def generate_response(message):
    best_matched_strings = similarity_search(message)
    # Prepare the best matched document string
    document_string = "\n".join(best_matched_strings)

    # Constructing the prompt
    template_string = """
    Please analyze the following best matched results from a similarity search, the user asked '{search_query}':

    {best_matched_meta_data}

    Please generate a summary from the best matched results.
    """

    # Create a PromptTemplate object
    prompt_template = PromptTemplate.from_template(template_string)

    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
    chain = LLMChain(llm=llm, prompt=prompt_template)

    response = chain.predict(best_matched_meta_data=document_string, search_query=message)
    return response


def main():
    st.set_page_config(page_title="Github wiki knowledge base", page_icon=":bird:")
    st.header("Github wiki knowledge base :bird:")
    message = st.text_area("Enter your question here")
    if message:
        st.write("Generating best knowledge response message...")
        result = generate_response(message)

        st.info(result)

if __name__ == "__main__":
    main()
