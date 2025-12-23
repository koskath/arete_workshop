from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch
import threading
import queue
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
tokenizer = None
model = None
model_name = "meta-llama/Llama-3.2-1B-Instruct" # our model is Koskath/arete-qwen-3b but it's really heavy at the moment I will make it available later for smaller machines
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
embeddings_model_name = "Qwen/Qwen3-Embedding-0.6B"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={"device": device})


# Adjust based on the course
with open("ml_ta.md", "r", encoding="utf-8") as f: # or sc_ta.md
    system_message = f.read()
db_name = "../vectorstore_creation/ml_vstore" # or "../vectorstore_creation/sc_vstore"
vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)


load_dotenv()


def retrieve_relevant_context(message, k=5):
    hits = vectorstore.similarity_search_with_relevance_scores(message, k=k)
    relevant_docs = []
    for doc, sim in hits:
        if sim > 0.5:
            relevant_docs.append(doc.page_content)
    
    if relevant_docs:
        relevant_context = "\n\n".join(relevant_docs)
        relevant_context += f"\n\nUser's Question: {message}"
        return relevant_context
    else:
        return message

class QueueStreamer(TextStreamer):
    def __init__(self, q, tokenizer, skip_prompt=True, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.q = q
        self.eos_token = tokenizer.eos_token

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if text == self.eos_token:
            self.q.put(None)
            return
        self.q.put(text)
        if stream_end:
            self.q.put(None)

def load_model_and_tokenizer(model_name = model_name):
    """Loads the model and tokenizer."""
    global tokenizer, model, device
    device_obj = torch.device(device)
    print(f"Using {device} device.")

    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    try:
        model = torch.compile(model, 
                            #   mode="reduce-overhead"
                            backend="aot_eager")
        print("Model compiled successfully.")
    except Exception as e:
        print(f"torch.compile failed: {e}. Running unoptimized.")
    model.to(device_obj)
    print("Model and tokenizer loaded.")




def generate_response(user_message, conversation_history):
    """Generate a response to the user's message."""
    retrieved_message = retrieve_relevant_context(user_message)
    temp_history = conversation_history + [{"role": "user", "content": retrieved_message}]

    text = tokenizer.apply_chat_template(
        temp_history,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(torch.device(device))

    q = queue.Queue()
    streamer = QueueStreamer(q, tokenizer, skip_prompt=True)
    
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=600,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_response = []
    print("\nAssistant: ", end="", flush=True)
    while True:
        token = q.get()
        if token is None:
            break
        full_response.append(token)
        print(token, end="", flush=True)

    thread.join()
    print()  # New line after response

    agent_response = "".join(full_response)
    # Update conversation history
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": agent_response.strip()})
    
    return agent_response.strip()

def main():
    """Main terminal-based chat loop."""
    global tokenizer, model
    
    print("Loading model...")
    load_model_and_tokenizer()
    
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