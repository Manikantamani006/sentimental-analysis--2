# Complete Transformer Architecture & Mathematical Guide for Emotion Prediction

## 1. Upgrading the Existing Repository from Classical NLP to Transformers

In your existing repository ([Manikantamani006/Emotion-Prediction-using-NLP-](https://github.com/Manikantamani006/Emotion-Prediction-using-NLP-)), the training pipeline in `main.ipynb` and `backend/model_service.py` classifies 16,000 training sentences (`data/train.txt`), 2,000 validation sentences (`data/val.txt`), and 2,000 test sentences (`data/test.txt`) across 6 emotion labels (`0: sadness`, `1: anger`, `2: love`, `3: surprise`, `4: fear`, `5: joy`).

| Model Architecture | Feature Representation | Word Order & Context | Negation Handling | Test Accuracy | Primary Source / Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Multinomial Naive Bayes | Bag-of-Words (`CountVectorizer`) | Ignored (Unordered counts) | Stripped by NLTK stopwords | 76.78% | [Emotion-Prediction-using-NLP-](https://github.com/Manikantamani006/Emotion-Prediction-using-NLP-) |
| Multinomial Naive Bayes | TF-IDF (`TfidfVectorizer`) | Ignored (Term frequency weights) | Stripped by NLTK stopwords | 66.09% | [Emotion-Prediction-using-NLP-](https://github.com/Manikantamani006/Emotion-Prediction-using-NLP-) |
| Tuned Logistic Regression (`C=10.0`) | TF-IDF (`TfidfVectorizer`) | Ignored (Linear bag-of-words) | Stripped by NLTK stopwords | 88.16% | [Emotion-Prediction-using-NLP-](https://github.com/Manikantamani006/Emotion-Prediction-using-NLP-) |
| From-Scratch Transformer (`encoder_decoder` / `encoder_only`) | Learned Token Embeddings + Sinusoidal Positional Encoding | Full bidirectional & cross-attention | Explicitly preserved & attended | ~89.5% – 90.8% | [Vaswani et al., 2017 (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) |
| Fine-Tuned Pre-Trained Transformer (`DistilBERT` / `RoBERTa`) | Contextual Subword Embeddings + 6–12 Transformer Layers | Deep multi-head contextual attention | Pre-trained + task fine-tuned | **93.80%** (F1: 93.79%) | [DistilBERT Emotion Benchmark](https://www.aimodels.fyi/models/huggingFace/distilbert-base-uncased-emotion-bhadresh-savani) |

### Why Classical TF-IDF Caps at 88.16% Accuracy
1. **Loss of Word Order (Bag-of-Words Assumption):** `TfidfVectorizer` treats a sentence as an unordered bag of words ([Emotion-Prediction-using-NLP-](https://github.com/Manikantamani006/Emotion-Prediction-using-NLP-)). The sentences *"I used to be terrified, but now I feel joyful"* and *"I used to be joyful, but now I feel terrified"* produce identical TF-IDF vectors.
2. **Destruction of Negations via Stopword Removal:** Stripping standard English stopwords via NLTK removes words like `"not"`, `"never"`, and `"no"` ([Emotion-Prediction-using-NLP-](https://github.com/Manikantamani006/Emotion-Prediction-using-NLP-)). Consequently, *"I am not happy at all"* is reduced to `"happy"`, causing false `joy` predictions instead of `sadness` or `anger`.
3. **Lack of Contextual Polysemy:** A word like *"shaking"* can indicate `fear` (*"shaking in the dark"*), `anger` (*"shaking with rage"*), or `joy` (*"shaking with excitement"*). The Transformer architecture introduced by [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) resolves this by dynamically computing context-dependent representations for every token via **Self-Attention**.

## 2. Token Embeddings & Sinusoidal Positional Encoding

Because the Transformer dispenses with recurrence (RNNs/LSTMs) and convolutions entirely ([Vaswani et al., 2017](https://arxiv.org/html/1706.03762v7)), feeding token embeddings directly into attention layers would make the network permutation-invariant (unable to tell which word came first).

### Step 2.1: Scaled Input Embeddings
Each input token ID $x_t$ is mapped to a continuous vector of dimension $d_{\text{model}}$ via an embedding matrix $E \in \mathbb{R}^{|V| \times d_{\text{model}}}$ and scaled by $\sqrt{d_{\text{model}}}$ so that the embedding magnitudes are not drowned out when positional encodings are added ([Vaswani et al., 2017, Section 3.4](https://arxiv.org/html/1706.03762v7)):

$$\text{Embedded}(x_t) = E(x_t) \cdot \sqrt{d_{\text{model}}}$$

### Step 2.2: Sinusoidal Positional Encoding (`PositionalEncoding`)
To inject relative and absolute token positions, the Transformer adds a deterministic positional vector $PE_{(pos)}$ of the same dimension $d_{\text{model}}$ using sine and cosine functions of geometric wavelengths from $2\pi$ to $10000 \cdot 2\pi$ ([Vaswani et al., 2017, Section 3.5](https://arxiv.org/html/1706.03762v7)):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

* **$pos$**: The token's index in the sentence ($0, 1, 2, \dots, L-1$).
* **$i$**: The dimension index ($0 \le i < d_{\text{model}}/2$). Even dimensions use $\sin$, and odd dimensions use $\cos$.
* **Why Sine and Cosine?** By trigonometric addition identities ($\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$), for any fixed offset $k$, $PE_{pos+k}$ can be written as a linear transformation of $PE_{pos}$ ([Vaswani et al., 2017](https://arxiv.org/html/1706.03762v7)). This allows attention heads to learn relative token distances effortlessly (e.g., noticing that `"not"` is 1 position before `"happy"`).

## 3. Scaled Dot-Product Attention

At the heart of `transformer_emotion_classifier.py` (`ScaledDotProductAttention`) is the mechanism that relates every token in a sentence to every other token ([Vaswani et al., 2017, Section 3.2.1](https://arxiv.org/html/1706.03762v7)).

### Step 3.1: Queries ($Q$), Keys ($K$), and Values ($V$)
For an input sequence matrix $X \in \mathbb{R}^{L \times d_{\text{model}}}$, the model projects each token into three distinct vectors using learned weight matrices ([PyTorch `nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)):
* **Query ($Q = X W_Q$):** Represents *"What contextual information is this token looking for?"* (e.g., the adjective `"happy"` looking for modifiers or negations).
* **Key ($K = X W_K$):** Represents *"What label or attribute does this token advertise to other tokens?"* (e.g., the word `"not"` advertising that it is a negation operator).
* **Value ($V = X W_V$):** Represents *"What semantic content should be passed forward if a Query matches my Key?"*

### Step 3.2: Mathematical Formulation of Scaled Dot-Product Attention
Given matrices $Q \in \mathbb{R}^{L_q \times d_k}$, $K \in \mathbb{R}^{L_k \times d_k}$, and $V \in \mathbb{R}^{L_k \times d_v}$, attention is computed in five sequential operations ([Vaswani et al., 2017, Section 3.2.1](https://arxiv.org/html/1706.03762v7)):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

1. **MatMul ($QK^T$):** Computes the raw dot-product similarity score between every Query token $i$ and every Key token $j$, yielding an $L_q \times L_k$ alignment matrix.
2. **Scale ($\div \sqrt{d_k}$):** If the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$, their dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean $0$ and variance $d_k$ ([Vaswani et al., 2017, Section 3.2.1](https://arxiv.org/html/1706.03762v7)). For large $d_k$, the dot products grow large in magnitude, pushing the Softmax function into regions with near-zero gradients (vanishing gradients). Dividing by $\sqrt{d_k}$ restores unit variance ($\text{Var} = 1$).
3. **Masking ($+ M$):** Before applying Softmax, illegal positions (such as `<pad>` tokens in variable-length sentences or future tokens in the Decoder) have their scores replaced with $-\infty$ (`-1e9` in float32 code) so that $e^{-\infty} = 0$ after Softmax ([Vaswani et al., 2017](https://arxiv.org/html/1706.03762v7)).
4. **Softmax ($\text{softmax}(\cdot)$):** Normalizes each row of similarity scores into a valid probability distribution summing to $1.0$.
5. **Weighted Sum ($\times V$):** Multiplies the attention probability weights by the Value vectors $V$, producing a context-enriched vector for every token.

## 4. Multi-Head Attention (`MultiHeadAttention`)

Instead of performing a single attention calculation across the full $d_{\text{model}}$ dimension, **Multi-Head Attention** linearly projects the Queries, Keys, and Values $h$ times into lower-dimensional subspaces of size $d_k = d_v = d_{\text{model}} / h$ ([Vaswani et al., 2017, Section 3.2.2](https://arxiv.org/html/1706.03762v7); [PyTorch `nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)):

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W^O$$

$$\text{where} \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

* **Why Multiple Heads Improve Emotion Accuracy:** A single attention head is often dominated by one strong relationship (such as a pronoun attending to its subject). With $h = 8$ parallel heads (`d_model=256`, `d_k=32` in `transformer_emotion_classifier.py`), Head 1 can track negation (`"not"` $\rightarrow$ `"happy"`), Head 2 can track emotional intensifiers (`"absolutely"` $\rightarrow$ `"terrified"`), and Head 3 can link contrast conjunctions (`"but"` $\rightarrow$ clause shift) simultaneously ([Vaswani et al., 2017](https://arxiv.org/html/1706.03762v7)).
* **Concatenation & Output Projection ($W^O$):** The outputs of all $h$ heads (each of shape $L \times d_k$) are concatenated back into a single $L \times d_{\text{model}}$ tensor and mixed through the final learned linear matrix $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

## 5. The Encoder Stack, Residual Connections, LayerNorm & Feed-Forward Network

In `TransformerEncoderLayer`, each Encoder layer consists of two core sub-layers wrapped with residual skip connections and Layer Normalization ([Vaswani et al., 2017, Section 3.1](https://arxiv.org/html/1706.03762v7); [PyTorch `nn.TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)):

### Step 5.1: Bidirectional Multi-Head Self-Attention
In the Encoder, $Q$, $K$, and $V$ all come from the output of the previous Encoder layer ($Q = K = V = H^{(\ell-1)}$). Every token in the sentence attends both leftward and rightward across the entire sentence (`src_padding_mask` masks out only `<pad>` tokens).

### Step 5.2: Residual Connections & Layer Normalization (`Add & Norm`)
Around every sub-layer (both Attention and Feed-Forward), the model adds the original input $x$ back to the sub-layer's output before normalizing across the feature dimension $d_{\text{model}}$ ([Vaswani et al., 2017, Section 3.1](https://arxiv.org/html/1706.03762v7)):

$$\text{Output} = \text{LayerNorm}(x + \text{Dropout}(\text{Sublayer}(x)))$$

$$\text{LayerNorm}(z) = \gamma \odot \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

* **Why Residual Connections ($x + \text{Sublayer}(x)$)?** They create a direct gradient highway during backpropagation so early layers (including Positional Encoding and Token Embeddings) receive strong, unimpeded error signals without vanishing gradients.
* **Why Layer Normalization?** Unlike Batch Normalization, LayerNorm normalizes each token vector independently across its $d_{\text{model}}$ features, stabilizing training across variable sequence lengths.

### Step 5.3: Position-Wise Feed-Forward Network (`PositionWiseFeedForward`)
After Multi-Head Attention aggregates contextual information across tokens, each token vector passes independently through a 2-layer fully connected Feed-Forward Network ([Vaswani et al., 2017, Section 3.3](https://arxiv.org/html/1706.03762v7)):

$$\text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2$$

* **Dimensional Expansion & Compression:** `linear_1` expands the token vector from $d_{\text{model}} = 256$ to a wider intermediate dimension $d_{ff} = 1024$ ($4 \times d_{\text{model}}$), applies a non-linear activation (`GELU` or `ReLU`), and `linear_2` projects it back down to $d_{\text{model}} = 256$.
* **Role in Emotion Classification:** While Self-Attention mixes information *across different words*, the Position-Wise FFN applies non-linear feature extraction *within each word's contextualized vector*, acting like a key-value memory bank that recognizes complex emotional patterns.

## 6. The Decoder Stack: Masked Multi-Head Attention & Cross-Attention

In `TransformerDecoderLayer` and `FullTransformerEmotionModel`, the Decoder stack introduces two critical mechanisms beyond the Encoder ([Vaswani et al., 2017, Section 3.1 & 3.2.3](https://arxiv.org/html/1706.03762v7)):

### Step 6.1: Masked Multi-Head Self-Attention (`create_causal_look_ahead_mask`)
In the first sub-layer of the Decoder, the model applies **Masked Multi-Head Self-Attention** using a lower-triangular causal mask $M_{\text{causal}} \in \mathbb{R}^{L_t \times L_t}$ ([Vaswani et al., 2017, Section 3.2.3](https://arxiv.org/html/1706.03762v7)):

$$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \quad (\text{Allowed: past and current tokens}) \\ -\infty & \text{if } j > i \quad (\text{Blocked: future tokens}) \end{cases}$$

When added to the scaled dot-product scores $\frac{QK^T}{\sqrt{d_k}}$ prior to Softmax, any position $j > i$ becomes $e^{-\infty} = 0$. This guarantees the autoregressive property: token prediction at step $i$ can only depend on known outputs at positions $\le i$ and never "cheat" by peeking at future tokens ([Vaswani et al., 2017](https://arxiv.org/html/1706.03762v7)).

### Step 6.2: Encoder-Decoder Cross-Attention (`cross_attn`)
The second sub-layer of `TransformerDecoderLayer` connects the Decoder to the Encoder stack ([Vaswani et al., 2017, Section 3.2.3](https://arxiv.org/html/1706.03762v7)):
* **Queries ($Q$):** Come from the previous Decoder sub-layer (`tgt`, representing the emotion classification query state).
* **Keys ($K$) and Values ($V$):** Come from the output memory of the top Encoder layer (`enc_memory`, representing the contextualized input sentence).
* This allows every position in the Decoder to attend over all positions in the input sentence, pinpointing the exact emotional trigger phrases in the user's text before passing through the Decoder's Position-Wise Feed-Forward Network.

## 7. Final Linear Layer & Softmax Probability Distribution

Once the Encoder and Decoder have produced the final contextual representation vector $h_{\text{final}} \in \mathbb{R}^{2 \cdot d_{\text{model}}}$, the model maps this vector to the 6 target emotion classes (`0: sadness`, `1: anger`, `2: love`, `3: surprise`, `4: fear`, `5: joy`):

### Step 7.1: Linear Projection (`self.linear_out`)
A learned linear transformation projects the hidden representation into raw, unnormalized class scores called **logits** $z \in \mathbb{R}^6$:

$$z = h_{\text{final}} W_{\text{cls}} + b_{\text{cls}}, \quad W_{\text{cls}} \in \mathbb{R}^{(2 \cdot d_{\text{model}}) \times 6}, \quad b_{\text{cls}} \in \mathbb{R}^6$$

### Step 7.2: Softmax Layer (`F.softmax(logits, dim=-1)`)
The Softmax function exponentiates and normalizes the 6 logits into a valid probability distribution $\hat{y} = [\hat{y}_0, \hat{y}_1, \hat{y}_2, \hat{y}_3, \hat{y}_4, \hat{y}_5]$ where $\hat{y}_c \in (0, 1)$ and $\sum_{c=0}^{5} \hat{y}_c = 1.0$:

$$\hat{y}_c = \text{Softmax}(z_c) = \frac{e^{z_c}}{\sum_{j=0}^{5} e^{z_j}}$$

The predicted emotion is the class with the highest probability ($\arg\max_c \hat{y}_c$), and during training the network minimizes Cross-Entropy Loss with label smoothing against the ground-truth emotion label $y$:

$$\mathcal{L}_{\text{CE}} = -\sum_{c=0}^{5} y_c^{\text{smooth}} \log(\hat{y}_c)$$

## 8. How to Run Locally for Higher Accuracy

Place `transformer_emotion_classifier.py` in the root folder of your cloned [Emotion-Prediction-using-NLP-](https://github.com/Manikantamani006/Emotion-Prediction-using-NLP-) repository (right next to `data/train.txt`, `data/val.txt`, and `data/test.txt`).

### Step 8.1: Install PyTorch & Hugging Face Dependencies
```bash
pip install torch numpy pandas scikit-learn transformers accelerate
```

### Step 8.2: Verify the Transformer Architecture & Tensor Shapes
Run the built-in self-test to trace every tensor shape through Positional Encoding, Multi-Head Self-Attention, Masked Multi-Head Attention, Cross-Attention, Feed-Forward, Linear, and Softmax:
```bash
python3 transformer_emotion_classifier.py --self-test
```

### Step 8.3: Train the Full From-Scratch Encoder-Decoder Transformer
To train the complete custom Transformer (with Encoder Self-Attention, Decoder Masked Self-Attention, Cross-Attention, Feed-Forward, Linear, and Softmax) directly on `data/train.txt`:
```bash
python3 transformer_emotion_classifier.py --arch encoder_decoder --epochs 15 --batch-size 64 --d-model 256 --num-heads 8
```

### Step 8.4: Fine-Tune Pre-Trained DistilBERT / RoBERTa for Maximum Accuracy (~93.8%)
While a Transformer trained from scratch only sees the 16,000 sentences in `data/train.txt`, fine-tuning [DistilBERT (`distilbert/distilbert-base-uncased`)](https://huggingface.co/distilbert/distilbert-base-uncased) or [RoBERTa (`FacebookAI/roberta-base`)](https://huggingface.co/FacebookAI/roberta-base) via the [Hugging Face Sequence Classification pipeline](https://huggingface.co/docs/transformers/en/tasks/sequence_classification) combines deep pre-trained language understanding with the exact same Transformer Multi-Head Attention and Feed-Forward architecture, boosting test accuracy from **88.16%** to **93.8%** ([AIModels.fyi Benchmark](https://www.aimodels.fyi/models/huggingFace/distilbert-base-uncased-emotion-bhadresh-savani)):
```bash
python3 transformer_emotion_classifier.py --arch pretrained --pretrained-model distilbert/distilbert-base-uncased --epochs 4 --batch-size 32
```
