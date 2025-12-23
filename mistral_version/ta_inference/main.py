from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# Load environment variables (including MISTRAL_API_KEY)
load_dotenv(override=True)

# Initialize Mistral embeddings
embeddings_model_name = "mistral-embed"
embeddings = MistralAIEmbeddings(model=embeddings_model_name)

# Initialize Mistral chat model
model_name = "ministral-3b-latest"
llm = ChatMistralAI(model=model_name, temperature=0.7)

# Adjust based on the course
with open("ml_ta.md", "r", encoding="utf-8") as f: # or sc_ta.md
    system_message = f.read()
db_name = "../vectorstore_creation/ml_vstore" # or "../vectorstore_creation/sc_vstore"
vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)


def retrieve_relevant_context(message, k=5):
    hits = vectorstore.similarity_search_with_relevance_scores(message, k=k)
    relevant_docs = []
    for doc, sim in hits:
        if sim > 0.8:
            relevant_docs.append(doc.page_content)
    
    if relevant_docs:
        relevant_context = "\n\n".join(relevant_docs)
        relevant_context += f"\n\nUser's Question: {message}"
        return relevant_context
    else:
        return message





def generate_response(user_message, conversation_history):
    """Generate a response to the user's message."""
    retrieved_message = retrieve_relevant_context(user_message)
    
    # Convert conversation history to LangChain messages
    messages = []
    for msg in conversation_history:
        if msg["role"] == "system":
            messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # Add the retrieved context as a user message
    messages.append(HumanMessage(content=retrieved_message))
    
    # Stream the response
    full_response = []
    print("\nAssistant: ", end="", flush=True)
    
    for chunk in llm.stream(messages):
        # Extract content from the message chunk
        if hasattr(chunk, 'content'):
            content = chunk.content
        else:
            content = str(chunk)
        if content:
            full_response.append(content)
            print(content, end="", flush=True)
    
    print()  # New line after response
    
    agent_response = "".join(full_response)
    # Update conversation history
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": agent_response.strip()})
    
    return agent_response.strip()

def main():
    """Main terminal-based chat loop."""
    print(f"Using Mistral API with model: {model_name}")
    print("Vectorstore loaded and ready.")
    
    # Initialize conversation history
    conversation_history = [
        {"role": "system", "content": system_message},
    ]
    
    print("\n" + "="*60)
    print("Teaching Assistant Chat - Terminal Mode")
    print("Type 'quit' or 'exit' to end the conversation")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            generate_response(user_input, conversation_history)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")

if __name__ == '__main__':
    main()