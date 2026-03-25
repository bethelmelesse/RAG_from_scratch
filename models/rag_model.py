from rag.model.generator import Generator
from rag.model.retriever import Retriever


class RAG:
    """RAG class to retrieve documents given a query."""

    def __init__(
        self, query_model_name: str, context_model_name: str, generator_model_name: str
    ):
        """Initialize the RAG."""
        self.retriever = Retriever(
            query_model_name=query_model_name, context_model_name=context_model_name
        )

        self.generator = Generator(model_name=generator_model_name)

    def toekn_level_rag(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        similarity_type: str = "inner_product",
    ) -> str:
        """Generate the response to the query using the token model.

        1- Retreiver (p(zᵢ | x)): retreives the top K documents given the query.
        2- Generator (p(y | x, zᵢ)): produces a distribution for the next output token
            given the query and the document.
        3- The distribution is marginalized over the documents to get the final
            distribution - p(y | x) = ∑ᵢ p(y | x, zᵢ) p(zᵢ | x).
        4- The token with the highest probability is selected as the next output token.
        5- This process is repeated until the end of the sequence is reached.

        Args:
            query (str): Input query
            documents (list[str]): List of documents to retrieve
            top_k (int): Number of documents to retrieve
            similarity_type (str): Type of similarity metric

        Returns:
            str: The generated response to the query
        """
        # Step 1: Retrieve the documents given a query (p(zᵢ | x))
        retreived_docs, retrieved_doc_probs = self.retriever.retrieve(
            query=query,
            documents=documents,
            top_k=top_k,
            similarity_type=similarity_type,
        )  # retrieved_doc_probs shape: (top_k,)

        # Step 2: Generate the next token probabilities (p(y | x)) for each document
        output = self.generator.token_level_generation(
            query=query,
            retreived_docs=retreived_docs,
            retrieved_doc_probs=retrieved_doc_probs,
        )

        return


if __name__ == "__main__":
    # Define the query and documents
    query = "What is the capital of France?"
    documents = [
        "The capital of Germany is Berlin.",
        "The capital of France is Paris.",
        "The capital of Italy is Rome.",
        "The capital of Spain is Madrid.",
        "The capital of Portugal is Lisbon.",
        "The capital of Greece is Athens.",
        "The capital of Turkey is Ankara.",
        "The capital of Egypt is Cairo.",
        "The capital of Saudi Arabia is Riyadh.",
        "The capital of UAE is Abu Dhabi.",
    ]

    # Initialize the DPR embedder
    query_model_name = "facebook/dpr-question_encoder-single-nq-base"
    context_model_name = "facebook/dpr-ctx_encoder-single-nq-base"
    generator_model_name = "facebook/bart-large"

    sequence_rag = RAG(
        query_model_name=query_model_name,
        context_model_name=context_model_name,
        generator_model_name=generator_model_name,
    )
    sequence_rag.toekn_level_rag(
        query=query, documents=documents, top_k=5, similarity_type="inner_product"
    )
