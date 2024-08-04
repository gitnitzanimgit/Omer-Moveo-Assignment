from flask import Flask, request, render_template_string, send_from_directory
import gensim
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel
import gensim.downloader as api
import nltk
import nest_asyncio
import threading
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import os
import atexit

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

SEED = 42

# Load the LDA model from the current working directory
best_model = gensim.models.LdaModel.load('models/lda_model_final.gensim')

# Load the dictionary from the current working directory
id2word = Dictionary.load('models/dictionary_final.gensim')

# Load the corpus from the current working directory
full_corpus = MmCorpus('models/corpus_final.mm')

def get_most_related_word(words):
    # Filter out words not present in the model's vocabulary
    # Load the Google News word2vec model
    model = api.load('word2vec-google-news-300')
    valid_words = [word for word in words if word in model]
    if not valid_words:
        return "Unnamed"
    # Find the word most related to the given words using the model
    related_word = model.most_similar(positive=valid_words, topn=1)
    if related_word:
        return related_word[0][0]
    return "Unnamed"

def output_for_task_3(num_topics):
    best_model = LdaModel(corpus=full_corpus,
                              id2word=id2word,
                              num_topics=num_topics,
                              random_state=SEED,
                              chunksize=100,
                              passes=10,
                              alpha='asymmetric',
                              eta=0.7)

    term_ratios = {}
    cluster_names = {}
    cluster_counts = {i: 0 for i in range(num_topics)}
    cluster_top_terms = {}

    for i in range(num_topics):
        term_ratios[i] = {}
        topic_terms = best_model.get_topic_terms(i, topn=20)  # Limiting to top 20 terms for efficiency
        for term_id, term_prob in topic_terms:
            term = id2word[term_id]
            overall_occurrence = sum([freq for doc in full_corpus for id, freq in doc if id == term_id])
            term_ratios[i][term] = term_prob / overall_occurrence

    for topic, terms in term_ratios.items():
        sorted_terms = sorted(terms.items(), key=lambda item: item[1], reverse=True)
        top_5_terms = [term for term, _ in sorted_terms[:5]]
        cluster_name = get_most_related_word(top_5_terms)
        cluster_names[topic] = cluster_name
        cluster_top_terms[cluster_name] = top_5_terms

    # Count the number of documents in each cluster
    for doc in best_model[full_corpus]:
        topic = max(doc, key=lambda x: x[1])[0]
        cluster_counts[topic] += 1

    # Prepare the combined output format
    combined_output = {cluster_names[topic]: {'count': count, 'top_terms': cluster_top_terms[cluster_names[topic]]}
                       for topic, count in cluster_counts.items()}

    # Generate pyLDAvis visualization
    vis_data = gensimvis.prepare(best_model, full_corpus, id2word, n_jobs=1)
    pyLDAvis.save_html(vis_data, 'models/ldavis.html')

    return combined_output

def cleanup():
    if os.path.exists('models/ldavis.html'):
        os.remove('models/ldavis.html')

# Register the cleanup function to be called on exit
atexit.register(cleanup)

nest_asyncio.apply()
def Task_3():
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        output_str = ""
        if request.method == "POST":
            num_topics = request.form.get("num_topics")
            if num_topics:
                try:
                    num_topics = int(num_topics)
                    output = output_for_task_3(num_topics)
                    output_str_counts = "\n".join([f"{cluster_name}: {data['count']}" for cluster_name, data in output.items()])
                    output_str_terms = "\n".join([f"{cluster_name}: {data['top_terms']}" for cluster_name, data in output.items()])
                    output_str = f"Counts:\n{output_str_counts}\n\nTop Terms:\n{output_str_terms}"
                except ValueError:
                    output_str = "Please enter a valid number."

        return render_template_string("""
        <!doctype html>
        <title>Topic Model Analyzer</title>
        <h1>Enter Number of Topics</h1>
        <form method="post">
          <input type="text" name="num_topics" placeholder="Enter number of topics here" required>
          <input type="submit" value="Submit">
        </form>
        <div>
          <pre>{{ output_str|safe }}</pre>
        </div>
        <div>
          <a href="/ldavis">View pyLDAvis Visualization</a>
        </div>
        """, output_str=output_str)

    @app.route('/ldavis')
    def ldavis():
        return send_from_directory('models', 'ldavis.html')

    def run_app():
        app.run(port=5001)

    # Run the Flask app in a separate thread
    thread = threading.Thread(target=run_app)
    thread.start()

# Run the Task_3 function to start the Flask app
Task_3()
