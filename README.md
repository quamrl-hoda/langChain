
# LangChain Learning Journey  

##  Overview  

Hi there! 👋  
I’m **Quamrul Hoda**, a **B.Tech student in Artificial Intelligence and Machine Learning (AIML)**.  
This repository contains all my **LangChain learning projects, experiments, and practice scripts**.  

I’ve **completed my LangChain learning** and gained **hands-on practical experience** by building multiple small projects — including LLM-based apps, data loaders, and AI workflows.  
This repository reflects my progress, coding practice, and understanding of how to use LangChain effectively with **OpenAI** and **Hugging Face** models.

## What’s Inside  
This repository covers core LangChain concepts with practical implementations:

-  **LangChain Basics** — Setup and Introduction  
-  **Prompt Templates** — Building structured prompts  
-  **Chains** — Sequential & Conditional Chains  
-  **Agents** — Tools, Actions, and Decisions  
-  **Runnables** — Lightweight and modular pipelines  
-  **Document Loaders** — Handling files and web content  
-  **Embeddings & Vector Stores** — Semantic search and retrieval  
-  **API Integrations** — Connecting OpenAI & Hugging Face models  


##  Project Structure  

```

langChain/
│
├── 1.langchain_basics/            # LangChain Introduction
├── 2.chat_models/                 # Working with Chat Models
├── 3.prompt_templates/            # Prompt Engineering
├── 4.chains/                      # Sequential and Conditional Chains
├── 5.agents/                      # Agents and Tool Usage
├── 6.runnables/                   # Runnable Pipelines
├── 7.document_loaders/            # Loading and Parsing Documents
├── 8.embeddings/                  # Embedding and Vector Storage
│
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

##  Getting Started  

###  Clone the Repository  
git clone https://github.com/quamrl-hoda/langChain.git
cd langChain


###  Install Requirements

pip install -r requirements.txt


### Run an Example
python 4.chains/conditional_chain.py

### ⚠️ Environment Setup

Create a `.env` file in your project root and add your API keys (keep them secret!):
OPENAI_API_KEY=your_openai_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token

## Example: “Hello LangChain”

Here’s a simple LangChain example that uses an **LLMChain** to generate text from a prompt.

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create a prompt
prompt = PromptTemplate.from_template("What is LangChain and why is it useful?")

# Create a chain
chain = LLMChain(prompt=prompt, llm=llm)

# Run the chain
response = chain.run({})
print(response)
```

 **Output Example:**

```
LangChain is a framework for building applications powered by language models. 
It helps developers connect LLMs with data, tools, and APIs to create intelligent workflows.
```
##  Learning Outcome

Through this repository, I have:

*  Built and tested **LangChain components**
*  Integrated **OpenAI** and **Hugging Face** APIs
*  Learned to build **Agents, Chains, and Runnables**
*  Understood **Prompt Engineering and Context Chaining**
*  Gained confidence in developing **LLM-powered AI applications**



## 👨‍🎓 About Me

**Name:** Quamrul Hoda
 **B.Tech Student (AI & ML)**
   Passionate about **Generative AI, LLMs, and LangChain**
   Focused on building **AI-driven applications** and learning real-world tools



## Connect With Me

[![GitHub](https://img.shields.io/badge/GitHub-quamrl--hoda-black?logo=github)](https://github.com/quamrl-hoda)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/quamrul-hoda-1a4247285/) 

##  License

This repository is open for learning and educational purposes.
Feel free to explore, fork, and learn from my LangChain journey 



> *“Learning by doing is the fastest way to master AI.”* – **Quamrul Hoda**

