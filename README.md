# AI-Powered-Document-Question-Answering-System
This project implements an intelligent document question-answering system capable of processing various document formats and providing accurate responses to user queries using advanced natural language processing techniques.
Features

Support for multiple document formats (PDF, TXT, EPUB, DOCX)
Natural language processing powered by OpenAI's GPT-4
Efficient document chunking and embedding for improved retrieval
Contextual compression for enhanced relevance of retrieved information
Interactive command-line interface for user queries

Prerequisites

Python 3.7+
OpenAI API key

Installation

Clone this repository:
Copygit clone https://github.com/yourusername/ai-document-qa.git
cd ai-document-qa

Install the required packages:
Copypip install -r requirements.txt

Set up your OpenAI API key:

Create a .env file in the project root
Add your OpenAI API key: OPENAI_API_KEY=your_api_key_here



Usage

Place your document in the project directory or specify the path in the document_path variable in main().
Run the script:
Copypython main.py

Enter your questions about the document when prompted. Type 'quit' to exit.

Configuration

Adjust the chunk_size and chunk_overlap in the configure_retriever() function to optimize for your specific use case.
Modify the similarity_threshold in the EmbeddingsFilter to fine-tune the relevance of retrieved information.
Change the model_name in configure_chain() to use a different OpenAI model.

Project Structure

main.py: The main script containing the document loading, retrieval, and QA chain setup.
DocumentLoader: A class for loading different document types.
configure_retriever(): Sets up the document retrieval system.
configure_chain(): Configures the question-answering chain.

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

LangChain for the awesome framework
OpenAI for the powerful language models
HuggingFace for the embedding models
FAISS for efficient similarity search
