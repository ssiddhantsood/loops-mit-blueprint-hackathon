import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import spacy
from textblob import TextBlob
import textstat

# Load tokenizer, model, and spaCy language model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
nlp = spacy.load("en_core_web_sm")

def get_embeddings(sentences, batch_size=10):
    """Generate embeddings for a batch of sentences."""
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def calculate_syntactic_complexity(text):
    """Calculate syntactic complexity using spaCy."""
    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    total_length = len([token for token in doc if not token.is_punct])
    average_sentence_length = total_length / num_sentences if num_sentences > 0 else 0
    return average_sentence_length

def calculate_semantic_diversity(text):
    """Calculate semantic diversity using unique lemmas."""
    doc = nlp(text)
    lemmas = set([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    total_lemmas = len(lemmas)
    total_tokens = len([token for token in doc if not token.is_punct])
    diversity_score = total_lemmas / total_tokens if total_tokens > 0 else 0
    return diversity_score

def calculate_readability_scores(text):
    """Calculate readability scores using textstat."""
    fk_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    return fk_grade, gunning_fog

def analyze_sentence_bias(sentences):
    """Directly analyze each sentence for bias based on subjectivity."""
    biased_sentences = []
    subjectivities = []
    for sentence in sentences:
        analysis = TextBlob(sentence)
        subjectivity = analysis.sentiment.subjectivity
        if subjectivity > 0.5:  # Adjust this threshold as needed
            biased_sentences.append(sentence)
        subjectivities.append(subjectivity)
    overall_bias_score = np.mean(subjectivities) * 100
    return biased_sentences, overall_bias_score

def normalize_value(value, min_range, max_range):
    """Normalize a value to a 0 to 1 scale based on expected min and max values."""
    return (value - min_range) / (max_range - min_range)

def normalize_value(value, min_range, max_range):
    """Normalize a value to a 0 to 1 scale based on expected min and max values."""
    return (value - min_range) / (max_range - min_range)

def analyze_website_text(text, biasWeight):
    """Analyze the provided text from a website for bias, complexity, and readability, with normalized scores."""
    # Preprocess text: Split into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Directly analyze sentences for bias
    biased_sentences, overall_bias_score = analyze_sentence_bias(sentences)

    # Calculate complexity and readability, then normalize
    syntactic_complexity = normalize_value(calculate_syntactic_complexity(text), 0, 40)
    semantic_diversity = normalize_value(calculate_semantic_diversity(text), 0, 1)  # May already be in 0-1 range
    fk_grade = normalize_value(textstat.flesch_kincaid_grade(text), 0, 20)
    gunning_fog = normalize_value(textstat.gunning_fog(text), 0, 20)

    # Average normalized scores for overall readability
    overall_readability_score = (syntactic_complexity + semantic_diversity + fk_grade + gunning_fog) / 4
    # Normalize bias score to 0-1 range (assuming 0-100 range initially)
    normalized_bias_score = overall_bias_score / 100

    # Calculate overall heuristic score
    heuristic_overall_score = (normalized_bias_score * biasWeight) + (overall_readability_score * (1-biasWeight))
    return {
        'biased_sentences': biased_sentences,
        'overall_bias_score': overall_bias_score,
        'overall_readability_score': overall_readability_score,
        'syntactic_complexity' : syntactic_complexity,
        'semantic_diversity': semantic_diversity,
        'fk_grade': fk_grade,
        'gunning_fog': gunning_fog,
        'sortingHeuristic': heuristic_overall_score,
    }





# # Example usage
# text = "The quick brown fox jumps over the lazy dog. This statement, while simple, underscores the agility of foxes in comparison to dogs, potentially biasing readers towards a pro-fox stance. In today's fast-paced world, technology drives progress. However, the relentless pursuit of innovation often overlooks the ethical implications, leading to widespread debate among experts. Studies show that early exposure to technology enhances learning capabilities in children, though skeptics argue it diminishes attention span. The Flesch-Kincaid Grade Level for this article is aimed to be moderate, balancing complexity with readability."
# result = analyze_website_text(text)
# print("Biased Sentences:", result['biased_sentences'])
# print("Overall Bias Score:", result['overall_bias_score'])
# print("Overall Complexity Score:", result['overall_complexity_score'])
# print("Readability Scores:", result['readability_scores'])
