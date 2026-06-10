# About RAGSearch

RAGSearch is a small Retrieval-Augmented Generation demo. It indexes local text
and Markdown files, then answers questions about them using OpenAI models.

## How it works

When you ingest a folder, each document is split into overlapping chunks. Every
chunk is converted into an embedding vector and stored in a local numpy index.

When you ask a question, the question is embedded too. The system compares the
question vector against every chunk using cosine similarity and keeps the most
similar chunks. Those chunks are passed to the chat model as context, and the
model answers using only that retrieved context, citing its sources.

## Why use it

RAG lets a language model answer questions about your own private documents
without retraining the model. It is cheap, fast to set up, and keeps answers
grounded in real source material.
