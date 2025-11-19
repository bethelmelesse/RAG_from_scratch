import torch

from models.dpr_embedder import DPREmbedder
from utils.faiss_utils import faiss_index_builder_and_search


class Retriever:
    """Retriever component for RAG model using DPR encoders and FAISS search.

    This class handles the retrieval phase of RAG by:
    1. Embedding queries and documents using DPR
    2. Building FAISS index for efficient similarity search
    3. Retrieving top-k most relevant documents with relevance scores
    """

    def __init__(self, query_model_name: str, context_model_name: str, device: str):
        """Initialize the retriever with DPR encoders.

        Args:
            query_model_name (str):  HuggingFace model identifier for the query encoder
            context_model_name (str): HuggingFace model identifier for the context encoder
            device (str): Device to run models on ('cuda' or 'cpu').
        """
        self.device = device
        self.dpr_embedder = DPREmbedder(
            query_model_name=query_model_name,
            context_model_name=context_model_name,
            device=device,
        )

    def retrieve(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        similarity_type: str = "inner_product",
    ) -> tuple[list[str], torch.Tensor]:
        """Retrieve the most relevant documents for a given query.

        Args:
            query (str): The query string to search for.
            documents (list[str]): List of candidate document strings to search through.
            top_k (int): Number of top documents to retrieve. Must be <= len(documents).
                Default is 5.
            similarity_type: Type of similarity metric for FAISS index.
                Options: 'inner_product', 'cosine', 'l2'.
                Default is 'inner_product'

        Returns:
            tuple[list[str], torch.Tensor]: A tuple containing:
                - retrieved_docs (list[str]):
                    List of top-k documents ordered by relevance (most relevant first)
                - retrieved_doc_probs (torch.Tensor):
                    Tensor of shape (top_k,) containing probability distribution over the retrieved documents (sums to 1.0)

        Raises:
            ValueError: If top_k exceeds the number of documents.
        """
        # Validate input
        if top_k > len(documents):
            raise ValueError(
                f"top_k ({top_k}) cannot exceed number of documents ({len(documents)})"
            )

        # Step 1: Embed the documents and query
        doc_embeds = self.dpr_embedder.embed_documents(documents=documents)
        query_embeds = self.dpr_embedder.embed_query(query=query)

        # Step 2: Build Faiss index
        distances, indices = faiss_index_builder_and_search(
            doc_embeds=doc_embeds,
            query_embeds=query_embeds,
            top_k=top_k,
            similarity_type=similarity_type,
        )

        # Step 2.1: Reshape distances and indices to 1D
        # Shape: [top_k,] -> [top_k,]
        distances = distances.squeeze()
        indices = indices.squeeze()

        # Step 3: Convert distances to scores
        if similarity_type == "cosine" or similarity_type == "inner_product":
            # Higher scores = better match, use directly
            scores = distances
        elif similarity_type == "l2":
            # Lower distance = better match, so negate
            scores = -distances
        else:
            raise ValueError(f"Unsupported similarity_type: {similarity_type}")

        # Step 4: Convert scores to probabilities using softmax
        # Note: This normalizes over top-k documents only.
        retreived_doc_probs = torch.softmax(scores, dim=0)

        # Step 5: Get the retreived documents in order of similarity
        retreived_docs = [documents[i] for i in indices]

        return retreived_docs, retreived_doc_probs


def main():
    """Test the Retriever class with sample documents."""
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the DPR embedder
    query_model_name = "facebook/dpr-question_encoder-single-nq-base"
    context_model_name = "facebook/dpr-ctx_encoder-single-nq-base"

    retriever = Retriever(
        query_model_name=query_model_name,
        context_model_name=context_model_name,
        device=device,
    )
    retreived_docs, retreived_doc_probs = retriever.retrieve(
        query=query,
        documents=documents,
        top_k=5,
        similarity_type="inner_product",
    )

    print(f"\nQuery: {query}")
    for i in range(len(retreived_docs)):
        print(
            f"Top {i + 1} Retreived Document (Probability: {retreived_doc_probs[i]}): {retreived_docs[i]}"
        )
    print()


if __name__ == "__main__":
    main()
