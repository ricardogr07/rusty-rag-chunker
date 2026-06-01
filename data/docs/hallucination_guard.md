Hallucination Guard in rusty-rag-chunker

The ask command in rusty-rag-chunker has a two-layer hallucination guard that prevents
the LLM from generating answers from its training data when the corpus does not contain
relevant information.

Layer 1: Score Threshold

Before calling the LLM, the ask command checks the cosine similarity score of the top
retrieved chunk. If this score is below the minimum retrieval threshold, the command
prints "I don't have information about this in the knowledge base." and exits immediately
without making an API call to OpenAI.

The minimum score threshold is controlled by the RETRIEVAL_MIN_SCORE environment variable.
The default value is 0.50. A higher value blocks more queries; a lower value lets more
borderline queries through.

Cosine similarity ranges: genuinely relevant chunks typically score above 0.70.
Unrelated content such as cooking recipes scores below 0.20. The threshold of 0.45 to 0.50
is a conservative cutoff that catches clear mismatches while allowing legitimate partial
matches through.

Layer 2: System Prompt Grounding

If the top chunk score passes the threshold, the LLM is called with a strict system prompt
that instructs gpt-4o-mini to answer only from the provided context. The system prompt says:
"Answer ONLY using the information provided in the context. Do NOT use outside knowledge
or training data."

The system role is far more effective than a user-message instruction for preventing
hallucination. Using the system role is the recommended approach for grounding LLM answers.

If the context does not contain the answer, gpt-4o-mini is instructed to respond with
"I don't have information about this in the knowledge base."

Together, the two layers mean the ask command will either produce a grounded answer
from the retrieved context or say it does not know. It will not generate a hallucinated
response from training data.
