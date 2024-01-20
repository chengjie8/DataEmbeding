import json
import os
import csv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders.csv_loader import CSVLoader

from dotenv import load_dotenv



load_dotenv()
loader = CSVLoader(file_path="output.csv")
document = loader.load()
# # Load your JSON data

# with open('output.json', 'r') as file:
#     data = json.load(file)
#     print(data[0])

# # Convert JSON to CSV
# with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)

#     # Writing the header of CSV file
#     writer.writerow(['title', 'url', 'html'])

#     # Writing data rows
#     for item in data:
#         writer.writerow([item['title'], item['url'], item['html']])



embedding = OpenAIEmbeddings()
db = FAISS.from_documents(document, embedding)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_content_array = [doc.page_content for doc in similar_response]
    return page_content_array

results = retrieve_info("what is LHDI?")
print(results)


template = """
You are a senior engineer.
1. Response should be a summery of relevant information, try to find the most matched information

Below is a message I received from a user.
{message}
Here's a summary of the message:
{summary}

Please write the best response to the user
"""

prompt = PromptTemplate(
    input_variables=["message", "summary"],
    template=template,
)
llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    best_summary = retrieve_info(message)
    response = chain.predict(message=message, summary=best_summary)
    return response

# response = generate_response("what is VRO?")
# print(response)


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
