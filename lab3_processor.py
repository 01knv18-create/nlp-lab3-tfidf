import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# --- ЗАВДАННЯ 1: Геометрія текстів ---
corpus = [
    "розподілені обчислення аналізують великі дані за допомогою кластера",
    "розподілені обчислення аналізують великі дані великі дані кластера кластера аналізують",
    "дата інженер будує конвеєр обробки інформації в реляційній базі даних",
    "прогноз погоди обіцяє суттєве потепління та грозові дощі на вихідні"
]

vectorizer = TfidfVectorizer(smooth_idf=True)
tfidf_matrix = vectorizer.fit_transform(corpus)

df_cos = pd.DataFrame(cosine_similarity(tfidf_matrix), columns=[f"Doc_{i+1}" for i in range(4)], index=[f"Doc_{i+1}" for i in range(4)])
df_euc = pd.DataFrame(euclidean_distances(tfidf_matrix), columns=[f"Doc_{i+1}" for i in range(4)], index=[f"Doc_{i+1}" for i in range(4)])

print("=== РЕЗУЛЬТАТИ ЗАВДАННЯ 1 ===")
print("Матриця косинусної схожості:\n", df_cos.round(4))
print("\nМатриця евклідової відстані:\n", df_euc.round(4))

# --- ЗАВДАННЯ 2: FAQ-агент ---
faq_data = pd.DataFrame([
    {"question": "що таке перенавчання моделі машинного навчання", "answer": "Це стан, коли модель показує високу точність на навчальних даних, але низьку на тестах."},
    {"question": "як працює токенізація на рівні окремих субслів bpe", "answer": "Алгоритм ітеративно об'єднує найбільш часті пари символів або байтів."},
    {"question": "чому для порівняння текстів обирають косинусну схожість", "answer": "Косинусна міра не залежить від довжини документів, вимірюючи кут між векторами."},
    {"question": "у чому різниця між алгоритмами стемінгу та лематизації", "answer": "Стемінг грубо відсікає суфікси, а лематизація знаходить канонічну форму слова за словником."},
    {"question": "як зменшити розмірність розрідженої матриці tf idf", "answer": "Використовуйте метод усіченого сингулярного розкладу (TruncatedSVD / LSA)."}
])

class TfidfFaqAgent:
    def __init__(self, faq_dataset):
        self.faq_df = faq_dataset.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(smooth_idf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.faq_df['question'])
        
    def query(self, user_question):
        q_vec = self.vectorizer.transform([user_question])
        scores = q_vec.dot(self.tfidf_matrix.T).toarray()
        best_idx = scores.argmax()
        top_3_indices = scores.argsort()[-3:][::-1]
        return {
            "query": user_question,
            "best_question": self.faq_df.iloc[best_idx]['question'],
            "answer": self.faq_df.iloc[best_idx]['answer'],
            "score": round(scores[best_idx].item(), 4),
            "top_3": [(idx, self.faq_df.iloc[idx]['question'],round(scores[idx].item(), 4) for idx in top_3_indices]
        }

bot = TfidfFaqAgent(faq_data)
test_queries = [
    "поясніть простими словами що таке перенавчання",
    "чому треба використовувати косинусну схожість векторів",
    "як швидко зменшити перенавчання bpe моделей"
]

print("\n=== РЕЗУЛЬТАТИ ЗАВДАННЯ 2 ===")
for q in test_queries:
    res = bot.query(q)
    print(f"\nЗапит: '{res['query']}'")
    print(f"-> Обране питання: '{res['best_question']}' (Score: {res['score']})")
    print(f"-> Топ-3 кандидати (.argsort): {res['top_3']}")
