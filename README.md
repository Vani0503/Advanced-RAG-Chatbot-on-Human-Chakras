# Advanced-RAG-Chatbot-on-Human-Chakras
This project addresses advanced features in a RAG-based AI Chatbot such as semantic chunking, query re-writing by adding conversational memory in manual way, hybrid vector search, filter chunks based on similarity score, add confidence label based on retrieval score evaluation metrics, etc

#Why I Built This
My simple RAG chatbot worked well for direct questions. Ask “What is the crown chakra?” and it answers correctly. But the moment I asked a follow-up question, “How do I activate it?”, the system completely lost the thread. It had no memory of what “it” referred to. Every question was treated as if it were the first question ever asked.

That’s not a chatbot. That’s a search engine with a friendly interface.

So I set out to build a proper conversational RAG system. What followed was three days of building, breaking, diagnosing, and rebuilding, and the most valuable learning came from the failures, not the features that worked.

#What I Built
Full feature list of the advanced RAG chatbot:

Semantic chunking — chunks sized by meaning, not character count

Manual conversational memory — full chat history passed to LLM at answer time

Query rewriting — vague follow-ups resolved into standalone questions before retrieval

Hybrid search — FAISS semantic search (70% weight) + BM25 keyword search (30% weight)

Retrieval filtering — chunks below a similarity score of 0.3 are discarded

Confidence scoring — 🟢 High / 🟡 Medium / 🔴 Low shown on every answer

Source citations — the user can see which PDF chunks the answer came from

Guided onboarding — suggested questions so users know what to ask

The knowledge base is the same five PDFs from my previous project, covering human chakras, hypnotherapy, family systems, and behavioural resolutions.

#The Two Approaches to Conversational Memory
Before anything else, it’s worth understanding the two ways to solve conversational memory in RAG, because this distinction shaped every decision I made.

##Approach 1: Automated (ConversationalRetrievalChain)

LangChain’s built-in approach. When a vague follow-up is asked, it automatically rewrites the query, searches the knowledge base, and answers using retrieved chunks plus history. Sounds clean. Two problems: it breaks when LangChain releases library updates (I experienced this firsthand on Streamlit Cloud), and it’s a black box you can’t see inside or debug when something goes wrong.

##Approach 2: Manual memory

You manage everything yourself. Store every user and assistant message. Before each answer, build the full chat history as plain text. Pass that history to the LLM alongside the retrieved context so it can answer with full awareness of the conversation.

This is what I built. More code, but fully owned, version-stable, and completely transparent.

Critical distinction: Manual memory alone does NOT fix retrieval. When a user asks “how do I activate it?”, that raw, vague question still goes to FAISS unchanged. FAISS has no access to chat history; it searches literally and returns generic or wrong results.

The LLM may paper over this using the chat history and still produce a decent-sounding answer, but the answer is built on the wrong foundation. I call this silent degradation. The answer feels right, but the retrieval underneath it was wrong.

This is why query rewriting is needed. It’s not a replacement for manual memory; it’s a component added on top to fix the retrieval step specifically.

#The First Live Test: Where the Whitelist Failed
After building, I tested on Streamlit. Asked “tell me about the heart chakra” then “give me its summary in 30 words.”

Response: about hypnotherapy. Nothing about the heart chakra.

##Root cause: I had built query rewriting with a whitelist approach, detecting vague words like “it”, “this”, “that” and only rewriting when those words appeared. “Its” was not in the whitelist. Raw vague question went to FAISS. FAISS retrieved chunks about hypnotherapy because “summary” and “30 words” matched content elsewhere. LLM answered from the wrong chunks.

I could have added “its” to the whitelist. But that’s exactly the problem, it’s whack-a-mole. Language is infinite. A keyword list is finite. Users write “condense that”, “eli5”, “too long, what’s the gist”, “what about the other one”. No list will ever cover every variation.

##The Fix: Always Rewrite
I replaced the whitelist entirely with an always-rewrite approach using the last 3-4 messages of conversation history.

One critical instruction added to the rewrite prompt: “if the question already makes complete sense on its own, return it unchanged.”

##This means:

“What is the crown chakra?”: passes through unchanged, already a clear, standalone question

“How do I activate it?”: gets rewritten to “How do I activate the crown chakra?” using history

No keywords. No maintenance. No gaps. The LLM handles every variation automatically.

##Query Drift: A Risk I Had to Guard Against
Always-rewrite introduces a new problem: query drift.

In a long, messy conversation:

User: tell me about root chakra
User: tell me about crown chakra
User: what meditation works for it?
User: how long should I do that?
User: does it work for the other one too?
By the final question, “the other one” could mean the root chakra, the crown chakra, or the meditation. The rewriter might pick the wrong referent and send a confidently wrong query to FAISS.

##Guards:

Only use the last 3-4 messages, not the full history

Instruct LLM to return the query unchanged if it already makes complete sense on its own

The Deeper Problem: Whether to Search, Not Just How
Even with perfect query rewriting, some questions should never go to FAISS at all.

When a user asks “give me its summary in 30 words”, they want the previous answer condensed, not a new answer from the knowledge base. No amount of query rewriting fixes this because the problem is not how you search, it is whether you search.

The real solution is intent classification: classify the question before retrieval.

RETRIEVAL — user wants new information → rewrite query → search FAISS → answer

CONVERSATIONAL — user wants to work with something already discussed → skip FAISS → work from history

HYBRID — user needs both → rewrite → search FAISS → answer using retrieved chunks and history

This replaces keyword lists entirely. The LLM understands “condense that”, “eli5”, “too long” as CONVERSATIONAL without any of those words listed anywhere.

I didn’t build this in the current project. It’s the headline feature of my next build.

##Semantic Chunking: Sounded Good, Caused Real Problems
I used SemanticChunker instead of fixed-size chunking. The idea: group text by meaning rather than arbitrary character count. Semantically related sentences stay together.

In practice, it caused a critical failure.

My PDFs have clearly named chapters: “CORRECTIVE THERAPY”, “CIRCLE THERAPY”, “HYPNODRAMA”. These are distinct concepts. SemanticChunker saw them as related therapy concepts and merged them together. Semantically correct. Retrieval-destructive.

Result: a search for “corrective therapy” returned Hypnodrama and Circle Therapy chunks. The LLM correctly followed my prompt: “if the answer is not in the context, say you don’t know”, and said, “I don’t know.” RAGAS scored 0.0 across all metrics for that question.

A 0.0 score is not just weak retrieval. It means the pipeline completely failed to produce an answer.

The lesson: Semantic chunking works best on flowing prose. For structured documents with distinct named sections, fixed-size chunking with overlap preserves topic boundaries better. The right strategy depends entirely on your document structure.

I rebuilt the index with RecursiveCharacterTextSplitter at chunk_size=1000, chunk_overlap=200, same as my simple RAG project.

##Chunk counts after rebuild:

Simple RAG:   821 chunks, avg 633 chars
Advanced RAG: 684 chunks, avg 769 chars  (slightly different due to PDF loading path)
RAGAS Evaluation: Run 1 (Invalid)
I set up RAGAS in Google Colab to compare both pipelines on 10 test questions across four metrics: faithfulness, answer relevancy, context precision, and context recall.

##Run 1 results:

Metric              Simple RAG   Advanced RAG
Faithfulness           0.983        0.900     ❌ Worse
Answer Relevancy       0.814        0.743     ❌ Worse
Context Precision      0.950        0.797     ❌ Worse
Context Recall         0.917        0.863     ❌ Worse
Advanced RAG lost on every metric. My first instinct was to accept this and write it up. My second instinct was to check whether the evaluation itself was valid.

It wasn’t. Four problems:

###Problem 1: Different k values. Advanced RAG used k=5, simple RAG k=3. More candidates mean more noise. Unfair comparison.

###Problem 2: Semantic chunking. One question scored 0.0 across all metrics, pulling the entire average down significantly.

###Problem 3: history_text was empty for every question. Including the conversational follow-up. Conversational memory and query rewriting were never actually tested in Run 1. The entire comparison was testing basic retrieval only.

###Problem 4: One conversational question out of ten. Even with history fixed, advanced features had minimal impact on overall scores. The test set was designed to test simple retrieval.

All four together: Run 1 told me nothing real. I fixed everything and reran.

#RAGAS Evaluation: Run 2 (Valid)
After fixing all four issues:

Metric              Simple RAG   Advanced RAG
Faithfulness           1.000        0.875     ❌ Worse
Answer Relevancy       0.809        0.811     ➡️ Same
Context Precision      0.950        0.942     ➡️ Same
Context Recall         0.940        0.883     ❌ Worse
What each result means:

Faithfulness: 1.0 vs 0.875

Advanced RAG’s LLM occasionally added information not strictly in the retrieved chunks. Likely cause: query rewriting subtly shifts the semantic meaning of questions, which can cause the LLM to draw slightly beyond the retrieved context. A faithfulness score below 0.9 in a production product is a hallucination risk worth monitoring closely.

Answer Relevancy: tied

Same LLM, same prompt. When retrieval quality is similar, the LLM produces similarly relevant answers. Expected.

Context Precision: tied

Both pipelines retrieved mostly relevant chunks. BM25 did not significantly hurt or help precision on this test set.

Context Recall: 0.940 vs 0.883

The most important finding. With k=3, every retrieval slot is precious. BM25 occasionally displaced a relevant semantic result with a keyword-matched but contextually weaker chunk. The fix: increase k to 5 so BM25 is additive, or increase semantic_weight from 0.7 to 0.85 to reduce BM25’s noise contribution. Both are on my list.

The One Question Where Advanced RAG Won Clearly
Q9 was the only conversational question: “What are its affirmations?” with history showing the previous question was about the throat chakra.

Simple RAG:   "I believe in myself."  ← one affirmation

Advanced RAG: 
  - I believe in myself
  - I am open to communication
  - I am listening precisely and expressing adequately
  - I'm fine, I'm okay
Query rewriting correctly resolved “its” to “throat chakra” using the history. Advanced RAG gave a four-affirmation complete answer. Simple RAG sent the raw vague question to FAISS, got lucky with a general affirmation chunk, and returned one.

RAGAS scored both answers similarly.

This is a real limitation of automated evaluation. RAGAS measures whether the answer is grounded in the retrieved context. It does not measure whether the answer is complete or useful. Simple RAG gave one affirmation. Advanced RAG gave four. From a user perspective, the difference is meaningful. RAGAS couldn’t see it.

User feedback — thumbs up/down — would capture this. Real users would prefer the complete answer.

#Key Takeaways
1. More features don’t automatically mean better metrics

Advanced RAG lost on two metrics in the valid comparison. Features add complexity. Complexity creates new failure modes. Every feature needs measurable evidence to earn its place, not intuition.

2. Your evaluation is only as valid as your test design

Run 1 said advanced RAG was worse. Run 2 said it was mostly comparable with specific gaps. Same systems, completely different conclusions, depending on how the evaluation was set up. Always verify your evaluation before concluding it.

3. Semantic chunking is not universally better

Depends entirely on the document structure. Flowing prose: semantic chunking preserves meaning. Structured documents with named sections: fixed-size chunking preserves topic boundaries better.

4. BM25 hurts recall when k is small

Hybrid search is supposed to be additive. With k=3, it becomes competitive, BM25 fights semantic results for the same three slots. Increase k before tuning weights.

5. RAGAS measures grounding, not completeness

A real limitation. Complement it with user feedback signals that capture whether the answer was actually useful, not just whether it was grounded.

6. Silent degradation is dangerous

The LLM can paper over weak retrieval using conversation history and produce a plausible-sounding answer. The answer feels right. The retrieval was wrong. Without evaluation, you’d never know. Build systems that fail loudly.

7. Your test set needs to test your actual features

Only 1 of 10 questions was conversational. Advanced RAG was built for conversational follow-ups and specific term searches. The test set was dominated by standalone questions where simple RAG excels. Evaluation design is a product decision.

#What’s Not Built Yet
Intent classification: every question still unconditionally triggers FAISS. A user asking “summarise that in 30 words” still hits the knowledge base even though it wants nothing from it. This is the architectural gap that query rewriting cannot fix. Next project.

More conversational test questions: 4-5 follow-up questions with populated history would properly validate what I built.

k=5 + semantic_weight=0.85: fixes the recall and precision gaps from the evaluation.

User feedback thumbs up/down: captures what RAGAS misses.

Chunk reranking: a second pass after retrieval, where a model scores chunks for relevance rather than similarity. Similarity and relevance are not the same thing.

20-25 test questions: 10 is too small for stable RAGAS scores. More questions covering diverse topics across all five PDFs.

#What This Taught Me as a PM
Evaluation design is a product decision. What you measure shapes what you build. A test set that doesn’t test your actual features produces misleading metrics that send you in the wrong direction. A PM who doesn’t understand their evaluation methodology cannot interpret their own product data.

Every feature needs a measurable hypothesis before you build it. Before adding hybrid search, the hypothesis should have been: “BM25 will improve context recall on Sanskrit term queries by X%.” Then run RAGAS before and after. I added it and hoped. Next time, I’ll define what winning looks like first.

Transparency is a product feature, not polish. Confidence scores, source citations, and suggested questions are trust signals. A user who can see where the answer came from is more likely to trust it and more likely to catch when it’s wrong. Hallucination liability isn’t just a technical problem. It’s a product problem.

Failing loudly is better than failing silently. When my system said “I don’t know” for corrective therapy, it looked like a failure. But it was the right behaviour, the system correctly refused to hallucinate when the retrieved context didn’t have the answer. Silent degradation, confident wrong answers are far worse. Design systems to fail visibly.

#What’s Next
To cover intent classification, routing questions before retrieval, so FAISS is only triggered when actually needed. I’ll measure whether it improves RAGAS scores and specifically whether it eliminates the “give me its summary” class of failures that query rewriting alone cannot fix.

