# 🎧 Audio & Text Emotion Detection 
This project is a robust emotion detection system that combines audio signal processing with natural language processing (NLP) to classify emotional states. It integrates classical ML techniques with advanced transformer models from Hugging Face, deployed through a FastAPI web server.

🔍 What This Project Does
Detects emotions from spoken audio input using Librosa (for feature extraction) and SVM (for classification).

Detects emotions from text input using two state-of-the-art transformer models:

j-hartmann/emotion-english-distilroberta-base

cardiffnlp/twitter-roberta-base-emotion

Offers a /predict_emotion endpoint via FastAPI to classify emotions with high confidence scores.

🧠 Models & Tools Used
🔉 Audio Emotion Detection
Librosa is used to extract features like MFCCs from audio samples.

A pre-trained Support Vector Machine (SVM) classifier is used to predict emotions from those features.

📝 Text Emotion Detection
Utilizes two powerful models from Hugging Face Transformers:

J-Hartmann's DistilRoBERTa: Fine-tuned for nuanced emotional understanding.

CardiffNLP RoBERTa: Specializes in emotion detection from social media text (Twitter, etc.).

Custom pre-processing functions handle casual, ambiguous, or mixed-intent user inputs to improve detection accuracy.

🧩 Logic & Heuristics
Pre-processing includes cleaning filler words, adding punctuation, and context enrichment.

Custom rule-based filters detect neutral and anger-related queries before passing them to models, enhancing interpretability and performance.

Post-processing adjusts outputs to reduce false positives for emotions like joy in clearly negative contexts.

🧪 API Features
Endpoint: /predict_emotion

Input: Text (e.g., transcribed audio or direct user input)

Output:

emotion: Detected emotion label

confidence: Model’s confidence score

text: Original input text

Returns clear status messages for both success and error cases.

🚀 Deployment
Runs locally using Uvicorn as the ASGI server.

Models load once at startup for efficiency.

Designed to be lightweight and deployable on cloud platforms.

📦 Dependencies
FastAPI

Transformers

Librosa

Scikit-learn

Uvicorn

Pydantic

Torch

Re (for regex-based preprocessing)

📈 Use Cases
Emotion-aware chatbots

Voice assistants

Mental health support tools

Customer feedback analysis

Sentiment analysis pipelines

⚙️ Extensibility
This project is modular and designed for expansion:

You can plug in other Hugging Face models or fine-tune your own.

The same logic could be extended to multilingual or domain-specific emotion classification.

📄 License
This project is open-source and available under the MIT License.

