import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import path from "path";
import { promises as fs } from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import {PromptTemplate} from "@langchain/core/prompts";
import {createStuffDocumentsChain} from "langchain/chains/combine_documents";
import {Ollama} from "@langchain/community/llms/ollama";
import {createRetrievalChain} from "langchain/chains/retrieval";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

// Create a new hono application
const app = new Hono()

// Ollama LLM configuration
const ollama = new Ollama({
  baseUrl: "http://localhost:11434", 
  model: "gemma2:2b",
});

// Embedding model configuration
const embeddings = new OllamaEmbeddings({
  model: "gemma2:2b",
  baseUrl: "http://localhost:11434",
  requestOptions: {
    useMMap: true,
    numThread: 6,
    numGpu: 1,
  },
});

// Method to read text file
const getTextFile = async () => {
  // Define file path
  const filePath = path.join(__dirname, "../data/langchain-test.txt");

  // Read the file and return data value
  const data = await fs.readFile(filePath, "utf-8");
  return data;
}

// Method to read pdf file
const loadPdfFile = async () => {
  // Define file path
  const filePath = path.join(__dirname, "../data/langchain-test.pdf");

 const loader = new PDFLoader(filePath);

  return await loader.load();
}

// Global variable for vector db
let vectorStore : MemoryVectorStore;

// Endpoint to load text embeddings
app.get('/loadTextEmbeddings', async (c) => {
  // Read text file
  const text = await getTextFile();

  // Text splitting settings
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    separators:['\n\n', '\n', ' ', '', '###'],
    chunkOverlap: 50
  });

  // Dividing text and creating documents
  const output = await splitter.createDocuments([text])

  // Define vector db
  vectorStore = await MemoryVectorStore.fromDocuments(output, embeddings);

  // Return success message
  const response = {message: "Text embeddings loaded successfully."};
  return c.json(response);
})

// Endpoint to load pdf embeddings
app.get('/loadPdfEmbeddings', async (c) => {
  // Read pdf file
  const documents = await loadPdfFile();

  // Define vector db
  vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);

  // Return success message
  const response = {message: "PDF embeddings loaded successfully."};
  return c.json(response);
})

// Endpoint to ask question
app.post('/ask',async (c) => {
  // Get question
  const { question } = await c.req.json();

  // Checking whether the vector database is loaded or not
  if(!vectorStore){
    return c.json({message: "Text embeddings not loaded yet."});
  }

  // Define a prompt template to tell bots role
  const prompt = PromptTemplate.fromTemplate(`You are a helpful AI assistant. Answer the following question based only on the provided context. If the answer cannot be derived from the context, say "I don't have enough information to answer that question." If I like your results I'll tip you $1000!

Context: {context}

Question: {question}

Answer: 
  `);

  // Define a document merge chain
  const documentChain = await createStuffDocumentsChain({
    llm: ollama,
    prompt,
  });

  // Define a retrieval chain
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever:vectorStore.asRetriever({
      k:3 // Get the 3 most similar documents
    })
  });

  // Processing the question and receiving the answer
  const response = await retrievalChain.invoke({
    question:question,
    input:""
  });

  // Converting answer to JSON format
  return c.json({answer: response.answer});
});

// Define server port number
const port = 3000
console.log(`Server is running on port ${port}`)

// Starting server
serve({
  fetch: app.fetch,
  port
})