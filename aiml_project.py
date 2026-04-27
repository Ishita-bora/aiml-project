import pandas as pd
from tkinter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv("career.csv")

X = data['text']
y = data['Career']

# -------------------------
# NLP MODEL
# -------------------------
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# -------------------------
# KEYWORD MAPPING
# -------------------------
keyword_map = {
    "coding": ["coding", "programming", "software", "developer"],
    "math": ["math", "logic", "problem solving"],
    "business": ["business", "finance", "marketing"],
    "communication": ["talking", "people", "social", "network"],
    "creative": ["art", "design", "makeup", "fashion", "styling"],
    "media": ["video", "youtube", "camera", "photography", "content"],
    "research": ["explore", "discover", "analyze", "learn"],
    "cooking": ["cooking", "chef", "food"]
}

def extract_features(text):
    text = text.lower()
    features = []

    for key, words in keyword_map.items():
        for word in words:
            if word in text:
                features.append(key)
                break

    return " ".join(features)

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict():
    user_input = text_input.get("1.0", END).strip()

    if user_input == "":
        result_label.config(text="⚠ Please enter something!")
        return

    processed = extract_features(user_input)

    if processed == "":
        processed = user_input  # fallback

    user_vec = vectorizer.transform([processed])
    probs = model.predict_proba(user_vec)[0]

    top3_idx = probs.argsort()[-3:][::-1]

    result = "🎯 Top Career Matches:\n\n"

    for i in top3_idx:
        career = model.classes_[i]
        percent = round(probs[i]*100, 2)
        result += f"{career} → {percent}%\n"

    # Reasoning
    if "media" in processed:
        result += "\n💡 You show interest in media & content creation"
    elif "creative" in processed:
        result += "\n💡 You are creative and expressive"
    elif "coding" in processed:
        result += "\n💡 You enjoy logic and problem solving"
    elif "business" in processed:
        result += "\n💡 You have business and communication skills"

    result_label.config(text=result)

# -------------------------
# GUI DESIGN (PINK THEME)
# -------------------------
root = Tk()
root.title("AI Career Guidance System")
root.geometry("650x500")
root.config(bg="#ffe6f0")  # light pink

title = Label(root,
    text="AI Career Guidance System",
    font=("Helvetica", 18, "bold"),
    bg="#ffe6f0",
    fg="#cc0066")
title.pack(pady=15)

subtitle = Label(root,
    text="Describe yourself (e.g., I like videos, fashion, coding)",
    font=("Helvetica", 11),
    bg="#ffe6f0",
    fg="#660033")
subtitle.pack()

text_input = Text(root,
    height=5,
    width=70,
    font=("Helvetica", 11),
    bg="#160f11")
text_input.pack(pady=10)

predict_btn = Button(root,
    text="Get Career Suggestions",
    font=("Helvetica", 12, "bold"),
    bg="#ba738f",
    fg="white",
    padx=12,
    pady=6,
    command=predict)
predict_btn.pack(pady=10)

result_label = Label(root,
    text="",
    font=("Helvetica", 11),
    bg="#cd9fb1",
    fg="#CD4B8C",
    justify=LEFT)
result_label.pack(pady=15)

root.mainloop()