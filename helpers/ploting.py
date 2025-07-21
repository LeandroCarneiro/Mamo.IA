from sklearn.metrics import confusion_matrix
import pandas as pd
import plotly.express as px

def display_confusion_matrix_pink_variants(estimator, X_test, y_test):
    # Get predictions
    y_pred = estimator.predict(X_test)
    
    # Create confusion matrix (normalized by true labels)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    # Convert to percentage
    cm = cm * 100
    
    # Create a dataframe for plotting
    classes = ['HEALTHY', 'PRE-BRCA', 'BRCA']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Create heatmap using plotly express
    fig = px.imshow(df_cm,
                    labels=dict(x="Predicted", y="True", color="Percentage"),
                    x=classes,
                    y=classes,
                    color_continuous_scale='RdPu',
                    aspect='auto',
                    title='Confusion Matrix (Normalized by True Labels)',
                    text_auto=True)  # This adds automatic text annotations
    
    # Customize the text format to show percentages
    fig.update_traces(texttemplate="%{z:.1f}%", textfont_size=12)
    
    # Update layout for better appearance
    fig.update_layout(
        width=500,
        height=400,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        font=dict(size=12)
    )
    
    # Ensure text is readable on all backgrounds
    fig.update_traces(textfont_color="white")
    
    return fig  