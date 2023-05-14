import streamlit as st
import openai
import PyPDF2
import numpy as np
import os




# Set up OpenAI API key
openai.api_type = "azure"
openai.api_base = "https://aks-production.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("KEY_AZURE_AI")

# Define function to extract text from PDF file
def extract_text(pdf_file):
    with pdf_file:
        read_pdf = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in read_pdf.pages:
            text += page.extract_text()
        return text


# Define function to generate embeddings for text
def generate_embeddings(text):
    model_engine = "dev03"  # can also use "davinci" for non-coding tasks
    prompt = f"summarize the document:\n{text}"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=2000,
        temperature=0.5
    )
    # embeddings = np.array(response.choices[0].embedding).astype(np.float32)
    embeddings = np.array(response.choices[0])
    return embeddings


# Define Streamlit app
def app():
    st.set_page_config(page_title="OpenAI PDF Reader")

    st.title("OpenAI PDF Reader")

    # Upload PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file:
        # Extract text from PDF file
        text = extract_text(pdf_file)

        # Generate embeddings for text using OpenAI API
        # embeddings = generate_embeddings(text)
        embeddings =text
        
        # Display text
        with st.expander("Show text"):
            st.write(text)
        
        # Ask a question using OpenAI API and compare against the generated embeddings
        question = st.text_input("What would you like to ask?")
        if st.button("Ask"):
            bot_context = "you are a bot that get doc context and users ask questions on the doc" + str(embeddings)
            response = openai.ChatCompletion.create(
                engine="gpt3",
                #prompt = 'f"role":"assistant","content": you are a bot that get doc context and users ask questions on the doc {embeddings} Q: {question} A:',
                messages=[{"role":"system","content":bot_context},
                  {"role":"user","content":question}
                  ],
            
                max_tokens=2000,
                stop=None,
                temperature=1
                )

            # Display answer
            print(question)
            answer =  str(response['choices'][0]['message']['content'])
#             answer = response.choices[0].text.strip()
            st.write(f"Answer: {answer}")


def main():
    app()


if __name__ == '__main__':
    main()
