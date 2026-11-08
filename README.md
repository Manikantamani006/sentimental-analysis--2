<div align="center">
  <img src="https://lh3.googleusercontent.com/aida/ALJJyrDz7hhbWSX4I5UiDpBfuWs6cMWHAIgY-kZCCnEmHZ5i8DPi0GmfR3Qp7W1WD9ccjNnMX-62x6loFOBBNlI4yo_KIBZoaNTYL-7RQzpsVmkSCT4r1jQ7ac9h_PTK0U9nlY700_grYtnSd1Zll2UBfl5uUBMuDRVY4F5Vlj0xvzzqmHCEwAH0sUgoPlomic2mc507LqZb50_Gq5zWUdJxvWRGdEPN5RDDWOVL_LOneBdH" alt="Sentima AI Logo" width="60"/>
  <h1>Sentima AI</h1>
  <p><b>DistilBERT Emotion & Sentiment Studio</b></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python version" />
    <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch" />
    <img src="https://img.shields.io/badge/FastAPI-0.110+-00a393.svg" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Design-Alexandria-6d5e00.svg" alt="Alexandria Design" />
  </p>
</div>

<br/>

![Sentima AI Dashboard](screen.jpg)

**Sentima AI** shifts sentiment analysis away from classical bag-of-words approaches into context-aware deep learning. By leveraging Multi-Head Self-Attention, the engine preserves word order, resolves negation traps (e.g., *"I am not happy"*), and outputs a granular 6-class emotional taxonomy.

The frontend utilizes the **Alexandria Design System**—a glassmorphic, editorial-grade UI prioritizing generous whitespace, serif authority, and tonal elevation.

---

## 📊 Performance Benchmark

Classical NLP models strip negations via stopwords. Sentima AI’s Transformer backbone attends to semantic structure, yielding a massive leap in test accuracy across 2,000 validation rows.

| Architecture | Accuracy | Context Handling |
| :--- | :--- | :--- |
| **Fine-Tuned DistilBERT** | `████████████████████` **93.80%** | Deep Self-Attention |
| **From-Scratch Transformer** | `██████████████████░░` **89.50%** | Global Cross-Attention |
| **Tuned Logistic Regression** | `████████████████░░░░` **88.16%** | TF-IDF (Linear) |
| **Naive Bayes (TF-IDF)** | `████████████░░░░░░░░` **66.09%** | Bag-of-Words |

---

## ⚙️ Inference Pipeline

```mermaid
graph LR
    A[Raw Text Input] -->|POST /api/analyze| B(FastAPI Server)
    B --> C{DistilBERT Engine}
    C -->|Tokenizer| D[Contextual Embeddings]
    D -->|Self-Attention| E[Feature Extraction]
    E --> F[6-Class Softmax]
    F -->|JSON Telemetry| G((Alexandria UI))
    
    style B fill:#094cb2,stroke:#000,stroke-width:1px,color:#fff
    style C fill:#3366cc,stroke:#000,stroke-width:1px,color:#fff
    style G fill:#6d5e00,stroke:#000,stroke-width:1px,color:#fff
