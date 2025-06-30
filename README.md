🤖 Customer Support AI Agent System
Welcome to the Customer Support AI Agent, an advanced AI system for automating ticket classification, response generation, and escalation decisions — optimized for Ubuntu Linux! 🎉

This project uses LangChain with a local LLM (e.g., Ollama) to process customer support tickets in real time.

📌 Features
✅ Ticket classification into categories (billing, technical, account access, etc.)
✅ Automatic priority assignment (low, medium, high, critical)
✅ Intelligent response generation with a professional tone
✅ Decision-making to auto-respond or escalate to a human agent
✅ Conversation memory tracking per ticket
✅ Integration with local LLMs (Ollama or LlamaCpp)
✅ Tools for updating ticket status, retrieving history, and searching a knowledge base
✅ Optimized for Ubuntu (system monitoring, logging, signal handling)
✅ Beautiful colored terminal outputs 🌈

🛠️ Requirements
Python 3.8+

Ollama installed and running with a pulled model (e.g., mistral:7b)

Virtual environment recommended

📦 Installation
bash
Copy
Edit
# Clone your project
git clone <your-repo-url>
cd <your-project-directory>

# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
✅ Install and Run Ollama
1️⃣ Install Ollama (if not done already):

bash
Copy
Edit
curl -fsSL https://ollama.com/install.sh | sh
2️⃣ Pull a model:

bash
Copy
Edit
ollama pull mistral:7b
3️⃣ Ensure Ollama is running:

bash
Copy
Edit
sudo systemctl start ollama
📄 Example Usage
Run your main script:

bash
Copy
Edit
python test1.py
🎫 The system will process sample tickets, classify them, decide on escalation, and generate responses — all shown in your terminal!

🔥 Project Structure
bash
Copy
Edit
├── test1.py           # Main script for the AI agent
├── requirements.txt   # Python dependencies
└── README.md          # This file 📝
🔧 Key Dependencies
LangChain (latest version recommended)

langchain-community (new home for many LangChain integrations)

langchain-ollama (if you switch to the new Ollama integration)

psutil (for system info)

colorama (for colored CLI output)
