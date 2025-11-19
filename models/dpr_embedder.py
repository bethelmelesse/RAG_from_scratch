import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)


class DPREmbedder:
    """DPR (Dense Passage Retrieval) embedder for RAG retriever component."""

    def __init__(self, query_model_name: str, context_model_name: str, device: str):
        """Initialize DPR embedder with query and context encoders.

        Args:
            query_model_name (str):  HuggingFace model identifier for the query encoder
            context_model_name (str): HuggingFace model identifier for the context encoder
            device (str): Device to run models on ('cuda' or 'cpu').
        """
        self.device = device

        # Initialize the tokenizers
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            query_model_name
        )
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            context_model_name
        )

        # Initialize models and move to device
        self.query_model = DPRQuestionEncoder.from_pretrained(query_model_name).to(
            device
        )
        self.context_model = DPRContextEncoder.from_pretrained(context_model_name).to(
            device
        )

        # Set models to evaluation mode
        self.query_model.eval()
        self.context_model.eval()

    def embed_query(self, query: str) -> torch.Tensor:
        """Embed a single query string into dense vector representation.

        Args:
            query (str): The query string to embed.

        Returns:
            torch.Tensor: Query embedding tensor of shape (1, embedding_dim)
        """
        with torch.no_grad():
            # Tokenize and move to device
            tokenized = self.query_tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            # Get embeddings
            query_embeddings = self.query_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).pooler_output

        return query_embeddings  # Shape: [1, embedding_dim]

    def embed_documents(self, documents: list[str]) -> torch.Tensor:
        """Embed a batch of document strings into dense vector representations.

        Args:
            documents (list[str]): List of document strings to embed.

        Returns:
            torch.Tensor: Document embeddings tensor of shape
                (num_documents, embedding_dim)
        """
        with torch.no_grad():
            # Tokenize and move to device
            tokenized = self.context_tokenizer(
                documents,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            # Get embeddings
            document_embeddings = self.context_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).pooler_output

        return document_embeddings  # Shape: [num_documents, embedding_dim]


def main():
    query_model_name = "facebook/dpr-question_encoder-single-nq-base"
    context_model_name = "facebook/dpr-ctx_encoder-single-nq-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    query = "What is the capital of France?"
    documents = ["The capital of France is Paris.", "The capital of Germany is Berlin."]

    # Initialize the DPR embedder
    dpr_embedder = DPREmbedder(
        query_model_name=query_model_name,
        context_model_name=context_model_name,
        device=device,
    )

    # Embed a query
    query_embeddings = dpr_embedder.embed_query(query=query)
    print(f"\nQuery: {query}")
    print(f"Query Embeddings: {query_embeddings.shape}")

    # Embed documents
    document_embeddings = dpr_embedder.embed_documents(documents=documents)
    print(f"\nDocuments: {documents}")
    print(f"Document Embeddings: {document_embeddings.shape}\n")


if __name__ == "__main__":
    main()
