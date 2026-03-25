"""Generator module for token-based RAG implementation."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Generator:
    """Generator class to generate the response to the query."""

    def __init__(self, model_name: str, device: str = None):
        """Initialize the generator."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def token_level_generation(
        self,
        query: str,
        retreived_docs: list[str],
        retrieved_doc_probs: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> torch.Tensor:
        """Token-level marginalized generation:
            p(y | x) = Σ_i p(y | x, z_i) p(z_i | x)


        Args:
            query (str): Input query
            retreived_docs (list[str]): List of retrieved documents
            retrieved_doc_probs (torch.Tensor): Probabilities of the retrieved documents
                    shape: (top_k,)
        Returns:
            torch.tensor: final sequence of tokens
        """
        inputs = [f"Question: {query}\nContext: {doc}" for doc in retreived_docs]

        tokenized_input = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        # Decoder starts with <bos> for each doc
        decoder_input_ids = torch.full(
            (len(retreived_docs), 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
        )

        # Make sure doc_probs is (k, 1)
        retrieved_doc_probs = retrieved_doc_probs.unsqueeze(1)

        generated_ids = []

        # ---- Autoregressive decoding ----
        for _ in range(max_new_tokens):
            out = self.model(
                input_ids=tokenized_input["input_ids"],
                attention_mask=tokenized_input["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )

            # logits: (k, seq_len, vocab)
            next_logits = out.logits[:, -1, :]
            next_probs = torch.softmax(next_logits, dim=-1)

            # ---- Marginalize documents ----
            # (k,1) * (k,vocab) -> sum → (vocab,)
            final_probs = torch.sum(retrieved_doc_probs * next_probs, dim=0)

            # Pick best token
            next_token_id = torch.argmax(final_probs).item()
            generated_ids.append(next_token_id)

            # Append this token to ALL decoder sequences
            next_tok = torch.tensor([[next_token_id]])
            next_tok = next_tok.repeat(len(retreived_docs), 1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_tok], dim=1)

            # Stop on EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
