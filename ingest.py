# ingest.py

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# A list of website URLs to scrape
WEBSITE_URLS = [
    "https://www.changiairport.com/in/en.html",
    "https://www.jewelchangiairport.com/"
]

def scrape_website_text(url: str) -> str:
    """
    Scrapes the text content from a given URL.
    It focuses on the main content area of the page.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text, use a space as a separator and strip leading/trailing whitespace
        text = soup.get_text(separator=" ", strip=True)
        
        # A simple cleanup to reduce multiple newlines and spaces
        cleaned_text = " ".join(text.split())
        
        print(f"✅ Successfully scraped {url}")
        return cleaned_text

    except requests.RequestException as e:
        print(f"❌ Failed to scrape {url}. Error: {e}")
        return ""

def main():
    """
    Main function to orchestrate the data ingestion pipeline.
    """
    print("🚀 Starting data ingestion process...")

    # --- 1. SCRAPING ---
    all_text = ""
    for url in WEBSITE_URLS:
        all_text += scrape_website_text(url) + "\n\n"

    if not all_text.strip():
        print("No text was scraped. Exiting.")
        return

    # --- 2. CHUNKING ---
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The max size of each chunk (in characters)
        chunk_overlap=150   # The number of characters to overlap between chunks
    )
    docs = text_splitter.split_text(all_text)
    print(f"Successfully split text into {len(docs)} chunks.")

    # --- 3. EMBEDDING & VECTOR STORE CREATION ---
    print("Generating embeddings and creating FAISS vector store...")
    
    # Use a Sentence-Transformer model from Hugging Face for embeddings
    # This runs locally on your machine
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Force running on CPU
    )

    # Create the FAISS vector store from the document chunks and embeddings
    # This can take a few moments depending on the amount of text
    vector_store = FAISS.from_texts(docs, embeddings)

    # --- 4. SAVING THE VECTOR STORE ---
    # Save the vector store locally. This file will be our knowledge base.
    vector_store.save_local("faiss_index")
    
    print("✅ All done! FAISS index created and saved locally in 'faiss_index' folder.")


if __name__ == "__main__":
    main()