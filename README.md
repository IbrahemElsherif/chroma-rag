# Chatbot with RAG and LangChain

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** chatbot using [LangChain](https://www.langchain.com/) and the OpenAI API. It allows the chatbot to answer questions based on your own documents by combining a vector database with an LLM.

---

## 📺 Full Tutorial

Watch the complete step-by-step guide on my YouTube channel:

[![Watch the tutorial](thumbnail_small.png)](https://www.youtube.com/watch?v=xf3gAFclwqo)

---

## 📋 Prerequisites

* **Python** 3.11 or higher
* An **OpenAI API Key** (Get it from [OpenAI](https://platform.openai.com/account/api-keys))

---

## 🚀 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ThomasJanssen-tech/Chatbot-with-RAG-and-LangChain.git
cd Chatbot-with-RAG-and-LangChain
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
```

### 3️⃣ Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Configure environment variables

* Rename `.env.example` → `.env`
* Add your OpenAI API key in the `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Running the Project

1. **Ingest your documents into the vector database**

```bash
python ingest_database.py
```

2. **Run the chatbot**

```bash
python chatbot.py
```

---

## 💡 Tips

* Make sure your documents are in the correct folder before running `ingest_database.py`.
* If you want to customize the prompt or retrieval strategy, check the `chatbot.py` file.
* You can change your shell prompt appearance with:

```bash
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

