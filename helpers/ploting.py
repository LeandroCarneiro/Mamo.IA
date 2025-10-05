from sklearn.metrics import confusion_matrix
import pandas as pd
import plotly.express as px

def display_confusion_matrix_pink_variants(estimator, X_test, y_test, classes):
    # Get predictions
    y_pred = estimator.predict(X_test)
    
    # Create confusion matrix (normalized by true labels)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    # Convert to percentage
    cm = cm * 100
    
    # Create a dataframe for plotting
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    fig = px.imshow(
        df_cm,
        text_auto='.1f',
        color_continuous_scale=[
            "#f8bbd0",  # light pink
            "#f06292",  # medium pink
            "#ad1457"   # dark pink
        ],
        aspect="auto",
        labels=dict(x="Predicted Label", y="True Label", color="Percentage"),
        title="Confusion Matrix (Pink Variants) - estimator: {}".format(estimator.__class__.__name__)
    )
    
    fig.show()
    
    # Optional: Return the figure for further customization or saving
    return fig


def display_confusion_matrix_pink_variants_mt_wt(estimator, X_test, y_test):
    # Get predictions
    y_pred = estimator.predict(X_test)
    
    # Create confusion matrix (normalized by true labels)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    # Convert to percentage
    cm = cm * 100
    
    # Create a dataframe for plotting
    classes = ['HEALTHY-MUT', 'HEALTHY-WT', 'BRCA']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    fig = px.imshow(
        df_cm,
        text_auto='.1f',
        color_continuous_scale=[
            "#f8bbd0",  # light pink
            "#f06292",  # medium pink
            "#ad1457"   # dark pink
        ],
        aspect="auto",
        labels=dict(x="Predicted Label", y="True Label", color="Percentage"),
        title="Confusion Matrix (Pink Variants) - estimator: {}".format(estimator.__class__.__name__)
    )
    
    fig.show()
    
    # Optional: Return the figure for further customization or saving
    return fig