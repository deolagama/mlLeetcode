import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("../data/raw/leetcode_raw.csv")
df = df[["id", "slug", "title", "difficulty", "content", "java", "c++", "python", "javascript"]]
df = df.melt(
    id_vars=["id", "slug", "title", "difficulty", "content"],
    value_vars=["java", "c++", "python", "javascript"],
    var_name="language",
    value_name="code"
)
df = df.dropna(subset=["code"]).reset_index(drop=True)
difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}
df["difficulty_num"] = df["difficulty"].map(difficulty_map)
df["content_length"] = df["content"].apply(lambda x: len(str(x).split()))
df["code_length"] = df["code"].apply(lambda x: len(str(x).split()))
le = LabelEncoder()
df["language_num"] = le.fit_transform(df["language"])

df.to_csv("../data/processed/leetcode_preprocessed.csv", index=False)

print("Preprocessed dataset saved:", df.shape,"✅")
print(df.head().to_string(index=False))

