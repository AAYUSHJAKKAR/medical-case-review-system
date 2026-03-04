# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB

# # 1. Load Data
# print("Loading Dataset...")
# df = pd.read_csv('dataset.csv')
# X = df['symptoms']
# y = df['disease']

# # 2. Preprocessing (Convert Text to Numbers)
# print("Vectorizing Text...")
# vectorizer = CountVectorizer()
# X_vectorized = vectorizer.fit_transform(X)

# # 3. Train Model (Naive Bayes)
# print("Training Model...")
# model = MultinomialNB()
# model.fit(X_vectorized, y)

# # 4. Save the "Brain" (Model) and "Translator" (Vectorizer)
# print("Saving Model to backend folder...")
# with open('../backend/model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# with open('../backend/vectorizer.pkl', 'wb') as f:
#     pickle.dump(vectorizer, f)

# # 5. Create a Remedy Lookup Dictionary
# remedy_dict = dict(zip(df['disease'], df['remedy']))
# with open('../backend/remedies.pkl', 'wb') as f:
#     pickle.dump(remedy_dict, f)

# print("✅ Success! Model is trained and saved in 'backend/' folder.")



import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- ROBUST PATH SETUP ---
# Get the folder where THIS script is currently located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the exact paths for the CSV and the Output folder
csv_path = os.path.join(current_dir, 'dataset.csv')
backend_dir = os.path.join(current_dir, '../backend')

# Ensure the backend directory actually exists before saving
if not os.path.exists(backend_dir):
    print(f"Creating missing backend directory at: {backend_dir}")
    os.makedirs(backend_dir)

# 1. Load Data
print(f"Loading Dataset from: {csv_path}")
if not os.path.exists(csv_path):
    print("❌ ERROR: dataset.csv not found! Check if you named it 'dataset.csv.txt'")
    exit()

df = pd.read_csv(csv_path)
X = df['symptoms']
y = df['disease']

# 2. Preprocessing (Convert Text to Numbers)
print("Vectorizing Text...")
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 3. Train Model (Naive Bayes)
print("Training Model...")
model = MultinomialNB()
model.fit(X_vectorized, y)

# 4. Save the "Brain" (Model) and "Translator" (Vectorizer)
print(f"Saving models to: {backend_dir}")

# Save the Model
model_path = os.path.join(backend_dir, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Save the Vectorizer
vectorizer_path = os.path.join(backend_dir, 'vectorizer.pkl')
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

# 5. Create a Remedy Lookup Dictionary
remedy_dict = dict(zip(df['disease'], df['remedy']))
remedy_path = os.path.join(backend_dir, 'remedies.pkl')
with open(remedy_path, 'wb') as f:
    pickle.dump(remedy_dict, f)

print("✅ Success! Models are trained and saved in 'backend/' folder.")