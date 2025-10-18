# RAG Chatbot – Quick Start

## How to Run

Make sure you have Python 3.11+ and have installed the dependencies:

```bash
pip install -r requirements.txt
```

1. Fill in your `.env` file with your OpenAI API key.

2. To launch the API server (FastAPI):

```bash
uvicorn api.main:app --reload
```

3. To launch the Gradio web demo:

```bash
python -m scripts.run_app
```

---

That's it!



## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

