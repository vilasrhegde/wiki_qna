import torch
import wikipedia
import transformers
import streamlit as st
from transformers import pipeline,Pipeline


def load_qa_pipeline() -> Pipeline:
    qa_pipeline = pipeline("question-answering",model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline


def load_wiki_summary(query: str)->str:
    results = wikipedia.search(query)
    summary = wikipedia.summary(results[0],sentences=10)
    return summary

def answer_question(pipeline:Pipeline,question: str,paragraph: str)->dict:
    input = {
        "question":question,
        "context":paragraph
    }
    output = pipeline(input)
    return output

if __name__=="__main__":    
    #title
    st.title("Wikipedia Question & Answers")
    st.write("Search Topic, Ask Question and Get Answer")

    #display topic input
    topic = st.text_input("SEARCH TOPIC","")

    article_paragraph = st.empty()

    question = st.text_input("QUESTION","")

    if topic:
        #load wiki summary of topic
        summary = load_wiki_summary(topic)
        #display summary
        article_paragraph.markdown(summary)

        # ask questions

        if question != "":
            
            qa_pipeline = load_qa_pipeline()

            # get the results for the question
            result = answer_question(qa_pipeline,question,summary)
            answer = result["answer"]

            #display answer
            st.write(answer)



