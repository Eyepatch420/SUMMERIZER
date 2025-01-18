from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import uvicorn
import os
import shutil

# Load environment variables
load_dotenv()

# Define the request body structure using Pydantic
class QuestionRequest(BaseModel):
    user_question: str

# Initialize FastAPI
app = FastAPI()

def get_pdf_text(pdf_files):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generates embeddings and saves them into a FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Sets up the conversational AI chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say "answer is not available in the context".\n\n
    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.post("/process-pdf/")
async def process_pdf(files: list[UploadFile] = File(...)):
    """
    API endpoint to upload and process PDF files.
    Extracts text, splits into chunks, and saves into a vector store.
    """
    try:
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded files temporarily
        file_paths = []
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_paths.append(file_path)

        # Extract text from PDFs
        raw_text = get_pdf_text(file_paths)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        # Clean up temporary files
        shutil.rmtree(temp_dir)

        return {"message": "PDFs processed and vector store created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {e}")

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    """
    API endpoint to answer user questions based on processed PDFs.
    """
    try:
        user_question = request.user_question

        # Load the vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Perform a similarity search for the user's question
        docs = vector_store.similarity_search(user_question)

        # Set up the conversational chain
        chain = get_conversational_chain()
        
        # Get the response from the chain
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        return {"response": response["output_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {e}")
if __name__ == "__main__":
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
