import faiss
import torch
import numpy as np


def faiss_index_builder_and_search(
    doc_embeds: torch.Tensor,
    query_embeds: torch.Tensor,
    top_k: int,
    similarity_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FAISS index for document embeddings and search for top-k similar documents.

    This function creates a FAISS index using the specified similarity metric,
    adds document embeddings to the index, and searches for the top-k most
    similar documents to the query.

    Args:
        doc_embeds (torch.Tensor): Document embeddings tensor of shape (num_docs, embedding_dim).
        query_embeds (torch.Tensor): Query embeddings tensor of shape (num_queries, embedding_dim).
            Typically num_queries=1 for single query retrieval.
        top_k (int): Number of top similar documents to retrieve.
        similarity_type (str): Similarity metric to use for search.
            - 'l2': Euclidean distance (lower is more similar)
            - 'cosine': Cosine similarity (requires L2-normalized embeddings)
            - 'inner_product': Inner product similarity (higher is more similar)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - distances: Tensor of shape (num_queries, top_k) containing similarity
              scores or distances. For 'inner_product' and 'cosine', higher values
              indicate more similar documents. For 'l2', lower values indicate more
              similar documents.
            - indices: Tensor of shape (num_queries, top_k) containing indices of
              the top-k documents in the original doc_embeds tensor.

    Raises:
        ValueError: If similarity_type is not one of ['l2', 'cosine', 'inner_product'].
    """
    device = doc_embeds.device

    # For cosine similarity, normalize embeddings
    if similarity_type == "cosine":
        doc_embeds_norm = torch.nn.functional.normalize(doc_embeds, p=2, dim=1)
        query_embeds_norm = torch.nn.functional.normalize(query_embeds, p=2, dim=1)
    else:
        doc_embeds_norm = doc_embeds
        query_embeds_norm = query_embeds

    # Move embeddings to CPU and convert to float32 numpy arrays
    doc_embeds_np = doc_embeds_norm.cpu().detach().float().numpy().astype(np.float32)
    query_embeds_np = (
        query_embeds_norm.cpu().detach().float().numpy().astype(np.float32)
    )

    # Get embedding dimension
    doc_embed_dim = doc_embeds_np.shape[1]

    # Build the FAISS index based on the similarity type
    if similarity_type == "l2":
        # L2 (Euclidean) distance: lower distance = more similar
        index = faiss.IndexFlatL2(doc_embed_dim)
    elif similarity_type == "cosine":
        # Cosine similarity via inner product (requires normalized embeddings)
        # Note: Caller must ensure embeddings are L2-normalized
        index = faiss.IndexFlatIP(doc_embed_dim)
    elif similarity_type == "inner_product":
        # Inner product similarity: higher score = more similar
        index = faiss.IndexFlatIP(doc_embed_dim)
    else:
        raise ValueError(
            f"Invalid similarity type: {similarity_type}. "
            f"Must be one of ['l2', 'cosine', 'inner_product']"
        )

    # Add document embeddings to the index
    index.add(doc_embeds_np)

    # Search the index for top-k similar documents
    # distances: similarity scores or distances. Shape: [1, top_k]
    # indices: indices of the top-k documents. Shape: [1, top_k]
    distances, indices = index.search(query_embeds_np, top_k)

    # Convert results back to PyTorch tensors on original device
    distances = torch.tensor(distances, device=device, dtype=torch.float32)
    indices = torch.tensor(indices, device=device, dtype=torch.int64)
    return distances, indices


def main():
    """Test FAISS index builder and search with sample embeddings."""
    # Create sample embeddings
    num_docs = 100
    num_queries = 2
    embedding_dim = 768

    # Random document and query embeddings
    doc_embeds = torch.randn(num_docs, embedding_dim)
    query_embeds = torch.randn(num_queries, embedding_dim)

    print("Testing FAISS Index Builder and Search")
    print(f"Document embeddings shape: {doc_embeds.shape}")
    print(f"Query embeddings shape: {query_embeds.shape}")

    # Test different similarity types
    for similarity_type in ["l2", "inner_product", "cosine"]:
        print(f"\n{similarity_type.upper()} Similarity:")

        # Search
        distances, indices = faiss_index_builder_and_search(
            doc_embeds=doc_embeds,
            query_embeds=query_embeds,
            top_k=5,
            similarity_type=similarity_type,
        )

        print(f"Distances shape: {distances.shape}")
        print(f"Indices shape: {indices.shape}")
        print(f"Top-5 indices for query 0: {indices[0].tolist()}")
        print(f"Top-5 scores for query 0: {distances[0].tolist()}")


if __name__ == "__main__":
    main()
