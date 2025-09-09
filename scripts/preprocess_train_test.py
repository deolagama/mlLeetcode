import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import load_dataset

ds = load_dataset("greengerong/leetcode")
df = ds["train"].to_pandas()
print("✅ Raw dataset loaded:", df.shape)
print(df.head())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("Training data:", train_df.shape)
print("Testing data:", test_df.shape)

def preprocess(df):
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

    return df

train_df_processed = preprocess(train_df)
test_df_processed = preprocess(test_df)

train_df_processed.to_csv("../data/processed/leetcode_train_preprocessed.csv", index=False)
test_df_processed.to_csv("../data/processed/leetcode_test_preprocessed.csv", index=False)

print("Preprocessed training data saved:", train_df_processed.shape)
print("Preprocessed testing data saved:", test_df_processed.shape)
