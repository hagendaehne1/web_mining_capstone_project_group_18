import os
import pandas as pd
from gpt4all import GPT4All
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()

# Initialize GPT4All model
model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# Load the data
path_neg = "./aclImdb_v1/aclImdb/test/neg"
path_pos = "./aclImdb_v1/aclImdb/test/pos"

files_neg = os.listdir(path_neg)
files_pos = os.listdir(path_pos)

number_of_samples_for_each_label = 25

data = []

# Read only the first 25 files
for file in files_neg[:number_of_samples_for_each_label]:
    file_path = os.path.join(path_neg, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f.name)
        data.append((f.read(), "negative"))
        
# Read only the first 25 files
for file in files_pos[:number_of_samples_for_each_label]:
    file_path = os.path.join(path_pos, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f.name)
        data.append((f.read(), "positive"))

# Create a Pandas DataFrame with columns 'text' and 'sentiment'
df = pd.DataFrame(data, columns=["text", "sentiment"])

# Define the labels for classification
candidate_labels = ['positive', 'negative']

# Prompt template with placeholder 0
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI model designed to perform sentiment analysis on movie reviews. Classify the sentiment of each review as either positive or negative, based on the tone, language, and overall opinion expressed in the text.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze the sentiment of the following movie review. The output must be only the label: '{0}'<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Function to classify each text using gpt4all
def classify_text_gpt4all(text):
    with model.chat_session(prompt_template=prompt_template):
        response = model.generate(prompt=text, max_tokens=10).strip()
        # For debugging, print response
        print(f"Model response: {response}")
        # Normalize the response to match the candidate labels
        response = response.lower()
        if 'positive' in response:
            return 'positive'
        elif 'negative' in response:
            return 'negative'
        else:
            return 'unexpected'  # Default fallback in case of unexpected output


# Apply the classifier to the 'text' column in the dataset
df["predicted_sentiment"] = df["text"].apply(classify_text_gpt4all)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(df["sentiment"], df["predicted_sentiment"])
precision = precision_score(df["sentiment"], df["predicted_sentiment"], average="weighted", zero_division=0)
recall = recall_score(df["sentiment"], df["predicted_sentiment"], average="weighted", zero_division=0)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Identify and display incorrect predictions
incorrect_predictions = df[df["sentiment"] != df["predicted_sentiment"]]
if not incorrect_predictions.empty:
    print("\nIncorrect Predictions:")
    print(incorrect_predictions[["text", "sentiment", "predicted_sentiment"]])
else:
    print("\nNo incorrect predictions!")


# Check unique values in ground truth and predictions for both cases
print("Ground truth labels (sentiment):", df["sentiment"].unique())
print("Predicted labels (GPT4All):", df["predicted_sentiment"].unique())

stop = timeit.default_timer()

print(f'Time: {(stop - start)/60} minutes')  

# Dynamically determine the unique labels in the dataset (When trying this code with too few test data so there is only one label (e.g. neutral), it tries to create a single tick, but the display_labels parameter tells it to expect three ticks. This mismatch leads to an error)
unique_labels = sorted(set(df["sentiment"]).union(set(df["predicted_sentiment"])))

# Plot confusion matrix using the dynamically determined labels
ConfusionMatrixDisplay.from_predictions(
    df["sentiment"],
    df["predicted_sentiment"],
    display_labels=unique_labels,
    cmap="Blues",
    colorbar=True
)
plt.title(f"Confusion Matrix (First {number_of_samples_for_each_label * 2} Samples)")
plt.show()

# Bar chart for accuracy, precision, and recall
metrics = ["Accuracy", "Precision", "Recall"]
values = [accuracy, precision, recall]

plt.bar(metrics, values, color=["blue", "orange", "green"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title(f"Evaluation Metrics (First {number_of_samples_for_each_label * 2} Samples)")
plt.show()