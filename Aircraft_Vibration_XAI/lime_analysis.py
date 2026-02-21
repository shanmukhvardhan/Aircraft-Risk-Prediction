import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def get_lime_explanation(model,scaler,input,feature_names,training_sample):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        feature_names = feature_names,
        training_data=scaler.transform(training_sample),
        mode = 'classification',
        class_names = ['Nominal','Risk']
    )
    scaled_input = scaler.transform(input)
    exp = explainer.explain_instance(
        data_row = scaled_input[0],
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    return exp
def plot_lime(explanation):
    fig = explanation.as_pyplot_figure()
    plt.tight_layout()
    return fig