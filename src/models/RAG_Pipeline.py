from transformers import pipeline

class PromptBuilder:
    def __init__(self, template):
        self.template = template

    def build(self, context_chunks, question):
        context_str = "\n\n".join(context_chunks)
        return self.template.format(context=context_str, question=question)
    

class LLMClient:
    def __init__(self, model_pipeline):
        self.generator = model_pipeline

    def generate(self, prompt, max_tokens=250):
        response = self.generator(prompt, max_new_tokens=max_tokens, do_sample=True)
        return response[0]["generated_text"].split("Answer:")[-1].strip()

class LLMClient_For_Llama:
    def __init__(self, llama_model):
        self.llm = llama_model

    def generate(self, prompt, max_tokens=250):
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.8,  # Equivalent to do_sample=True
            stop=["Answer:"]   # Helps split the output like your original code
        )
        return response["choices"][0]["text"].split("Answer:")[-1].strip()
    
class RAGAgent:
    def __init__(self, searcher, prompt_builder, llm_client):
        """
        Initialize the RAGAgent with searcher, prompt builder, and LLM client.

        :param searcher: An object responsible for executing search queries and returning relevant results.
        :param prompt_builder: An instance of PromptBuilder used to construct prompts with context and user queries.
        :param llm_client: A client interface to the language model for generating responses based on prompts.
        """

        self.searcher = searcher
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

    def run(self, user_query):
        
        """
        Executes a search and generation process for a given user query using the RAG architecture.

        :param user_query: A string representing the user's input query.
        :return: A string containing the generated response based on the retrieved context and user query.
        """

        results = self.searcher.search(user_query, top_k=5, return_full_text=True)
        chunks = results["chunk_text"].tolist()
        prompt = self.prompt_builder.build(chunks, user_query)
        return self.llm_client.generate(prompt)