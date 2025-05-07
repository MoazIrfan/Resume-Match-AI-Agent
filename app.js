import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { createOpenAIFunctionsAgent, AgentExecutor } from 'langchain/agents';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { DynamicTool } from '@langchain/core/tools';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from '@langchain/core/documents';
import * as fs from 'fs/promises';
import * as dotenv from 'dotenv';

dotenv.config();

// Read and split the resume into chunks
async function loadResumeChunks() {
  const content = await fs.readFile('documents/resume.md', 'utf8');
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const documents = await splitter.createDocuments([content]);
  return documents;
}

// Create vector store from resume chunks
async function createVectorStore(documents) {
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);
  return vectorStore;
}

// Define the ResumeRetriever tool
function createResumeRetrieverTool(vectorStore) {
  return new DynamicTool({
    name: 'ResumeRetriever',
    description:
      'Use this tool to fetch relevant experience, skills, or achievements from the resume based on job requirements. Use it only ONCE per task.',
    func: async (input) => {
      const results = await vectorStore.similaritySearch(input, 3);
      return results.map((doc) => doc.pageContent).join('\n---\n');
    },
  });
}

// Set up the Chat model
const model = new ChatOpenAI({
  modelName: 'gpt-4o',
  temperature: 0.3,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// Define the prompt template
const prompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    'You are a career assistant that writes job-tailored cover letters. If the user input includes a job description, use the ResumeRetriever tool ONCE to extract relevant experience/skills. Do not guess.',
  ],
  new MessagesPlaceholder('agent_scratchpad'),
  ['user', '{input}'],
]);

// Main execution function
async function main() {
  const documents = await loadResumeChunks();
  const vectorStore = await createVectorStore(documents);
  const resumeRetrieverTool = createResumeRetrieverTool(vectorStore);

  const agent = await createOpenAIFunctionsAgent({
    llm: model,
    tools: [resumeRetrieverTool],
    prompt,
  });

  const executor = new AgentExecutor({
    agent,
    tools: [resumeRetrieverTool],
    verbose: true,
    maxIterations: 5,
  });

  const input = `Write a cover letter tailored to the following job description:
  
  "We are looking for a skilled and passionate React.js Developer with a strong understanding of TypeScript, unit testing, and micro-frontend architecture. As part of our dynamic team, you will be responsible for designing, developing, and maintaining scalable, high-performance web applications, focusing on the frontend architecture and improving the user experience."
  `;

  const result = await executor.invoke({ input });
  console.log('\n📝 Final Cover Letter:\n');
  console.log(result.output);
}

main();
