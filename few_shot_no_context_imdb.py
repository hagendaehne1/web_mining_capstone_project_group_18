import os
import pandas as pd
from gpt4all import GPT4All
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()

# Initialize GPT4All model
model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", n_ctx=4096)

# # Load the data
path_neg = "./test/neg"
path_pos = "./test/pos"

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

You are a sentiment analysis bot for labeling text. Use the following labels to classify the sentiment:

Positive: The text expresses a favorable, optimistic, or happy sentiment.
Negative: The text expresses an unfavorable, pessimistic, or unhappy sentiment.
Please read the text carefully and assign one of the labels based on the overall sentiment conveyed.<|eot_id|><|start_header_id|>user<|end_header_id|>

Here are a few examples of texts and their sentiment:
###
Example 1:
Text: Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off putting. Even those from the era should be turned off. The cryptic dialogue would make Shakespeare seem easy to a third grader. On a technical level it's better than you might think with some good cinematography by future great Vilmos Zsigmond. Future stars Sally Kirkland and Frederic Forrest can be seen briefly.
Sentiment: negative

###
Example 2:
Text: Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as "Teachers". My 35 years in the teaching profession lead me to believe that Bromwell High's satire is much closer to reality than is "Teachers". The scramble to survive financially, the insightful students who can see right through their pathetic teachers' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High. A classic line: INSPECTOR: I'm here to sack one of your teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isn't!
Sentiment: positive

###
Example 3:
Text: Robert DeNiro plays the most unbelievably intelligent illiterate of all time. This movie is so wasteful of talent, it is truly disgusting. The script is unbelievable. The dialog is unbelievable. Jane Fonda's character is a caricature of herself, and not a funny one. The movie moves at a snail's pace, is photographed in an ill-advised manner, and is insufferably preachy. It also plugs in every cliche in the book. Swoozie Kurtz is excellent in a supporting role, but so what?<br /><br />Equally annoying is this new IMDB rule of requiring ten lines for every review. When a movie is this worthless, it doesn't require ten lines of text to let other readers know that it is a waste of time and tape. Avoid this movie.
Sentiment: negative

###
Example 4:
Text: If you like adult comedy cartoons, like South Park, then this is nearly a similar format about the small adventures of three teenage girls at Bromwell High. Keisha, Natella and Latrina have given exploding sweets and behaved like bitches, I think Keisha is a good leader. There are also small stories going on with the teachers of the school. There's the idiotic principal, Mr. Bip, the nervous Maths teacher and many others. The cast is also fantastic, Lenny Henry's Gina Yashere, EastEnders Chrissie Watts, Tracy-Ann Oberman, Smack The Pony's Doon Mackichan, Dead Ringers' Mark Perry and Blunder's Nina Conti. I didn't know this came from Canada, but it is very good. Very good!
Sentiment: positive

###
Example 5:
Text: Bromwell High is nothing short of brilliant. Expertly scripted and perfectly delivered, this searing parody of a students and teachers at a South London Public School leaves you literally rolling with laughter. It's vulgar, provocative, witty and sharp. The characters are a superbly caricatured cross section of British society (or to be more accurate, of any society). Following the escapades of Keisha, Latrina and Natella, our three "protagonists" for want of a better term, the show doesn't shy away from parodying every imaginable subject. Political correctness flies out the window in every episode. If you enjoy shows that aren't afraid to poke fun of every taboo subject imaginable, then Bromwell High will not disappoint!
Sentiment: positive

###
Example 6:
Text: I saw the capsule comment said "great acting." In my opinion, these are two great actors giving horrible performances, and with zero chemistry with one another, for a great director in his all-time worst effort. Robert De Niro has to be the most ingenious and insightful illiterate of all time. Jane Fonda's performance uncomfortably drifts all over the map as she clearly has no handle on this character, mostly because the character is so poorly written. Molasses-like would be too swift an adjective for this film's excruciating pacing. Although the film's intent is to be an uplifting story of curing illiteracy, watching it is a true "bummer." I give it 1 out of 10, truly one of the worst 20 movies for its budget level that I have ever seen.
Sentiment: negative

###
Label this. The output must be only the label: 
###

Text: {0}
Sentiment: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
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
