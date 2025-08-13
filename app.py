import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# load dataset
dataset = pd.read_csv("wine_quality_classifier.csv")

# create a condition means wine is good above 7
threshold = 7
dataset["fine"] = (dataset["quality"] >= threshold).astype(int)

# features and target
x = dataset.drop(["quality", "fine"], axis=1)
y = dataset["fine"]

# Balancing dataset
balance = RandomOverSampler()
x_balance, y_balance = balance.fit_resample(x, y)

# train and testing split
x_train, x_test, y_train, y_test = train_test_split(
    x_balance, y_balance, test_size=0.2, random_state=42
)

# Machine learning model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# accuracy
accuracy = model.score(x_test, y_test)
model_accuracy = f"{round(accuracy * 100, 2)} %"

# all column names
feature_names = list(x.columns)


# making function
def predict_quality(*args):
    user_input = np.array(args).reshape(1, -1)
    prediction = model.predict(user_input)[0]
    proba = model.predict_proba(user_input)[0]


#making pie chart
    fig, ax = plt.subplots()
    labels = ["Bad", "Good"]
    colors = ["red", "green"]
    ax.pie(proba, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.title("Ratio Distribution")

    if prediction == 1:
        return (
            "The Quality of Wine is 'GOOD' ✅",
            model_accuracy,
            fig,
        )

    else:
        return (
            "The Quality of Wine is 'BAD' ❌",
            model_accuracy,
            fig,
        )

#inputs
inputs = [
    gr.Slider(
        minimum=float(dataset[column].min()),
        maximum=float(dataset[column].max()),
        value=float(dataset[column].mean()),
        step=0.1,
        label=column,
    )
    for column in feature_names
]

#gradio Interface
demo = gr.Interface(
    fn=predict_quality,
    inputs=inputs,
    outputs=[
        gr.Label(
            label="Output",
        ),
        gr.Label(label="Accuracy"),
        gr.Plot(label="Graph", show_label=True),
    ],
    title="🍷 Wine Quality Classifier",
    description="A Wine Quality Classifier that predicts whether the wine is Good or Bad based on chemical features",
    article="MADE BY MOHD ALTAMASH",
)


demo.launch(share=True)
