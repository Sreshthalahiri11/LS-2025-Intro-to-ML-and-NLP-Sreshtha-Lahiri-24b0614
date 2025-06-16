import pandas as pd
import numpy as np

data = {
    "Name": [f"Student {i}" for i in range(1, 11)],
    "Subject": np.random.choice(["Math", "Physics", "Chemistry"], 10),
    "Score": np.random.randint(50, 101, 10),
    "Grade": [""] * 10,
}
df = pd.DataFrame(data)

df["Grade"] = pd.cut(df["Score"], bins=[0, 59, 69, 79, 89, 100], labels=["F", "D", "C", "B", "A"])
print("DataFrame with Grades:\n", df)

df_sorted = df.sort_values("Score", ascending=False)
print("DataFrame Sorted by Score:\n", df_sorted)

avg_scores = df.groupby("Subject")["Score"].mean()
print("Average Scores by Subject:\n", avg_scores)

def pandas_filter_pass(dataframe):
    return dataframe[dataframe["Grade"].isin(["A", "B"])]

filtered_df = pandas_filter_pass(df)
print("Filtered DataFrame (Grades A & B):\n", filtered_df)
