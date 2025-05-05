An interactive Streamlit web application developed for the AF3005 Programming for Finance course assignment. This application allows users to perform machine learning tasks (Linear Regression, Logistic Regression, K-Means Clustering) on financial data, integrating datasets from Kaggle (CSV).

The entire application is wrapped in a fun "Breaking Bad" chemistry lab theme, featuring custom styling, thematic elements, and step-by-step guidance through the ML workflow.


## Features ✨

*   **Thematic Interface:** Fully themed "Breaking Bad" chemistry lab aesthetic with green/yellow color schemes, custom CSS, and chemistry-related icons/GIFs.
*   **Data Sources:**
    *   Upload financial datasets in CSV format (e.g., from Kragle).
    *   Fetch real-time and historical stock data using Yahoo Finance (`yfinance`).
*   **Machine Learning Models:**
    *   Implement **Linear Regression** for prediction tasks.
    *   Implement **K-Means Clustering** for unsupervised grouping.
    *   Implement **Logistic Regression** for binary classification (e.g., predicting stock price direction).
    *   **Dynamic Model Selection:** Choose the desired ML "Formula" via a dropdown menu.
*   **Step-by-Step ML Pipeline:** Guided workflow with distinct stages:
    1.  **Load Data** ("Cook the Data")
    2.  **Preprocessing** ("Purify the Mixture") - Handles missing values.
    3.  **Feature Engineering/Selection** ("Select the Ingredients")
    4.  **Train/Test Split** ("Divide the Batch") - For supervised models.
    5.  **Model Training** ("Brew the Model")
    6.  **Evaluation** ("Test the Product") - Displays relevant metrics (RMSE, Accuracy, Silhouette Score).
    7.  **Results Visualization** ("Inspect the Final Batch") - Interactive Plotly charts.
*   **Interactive Visualizations:** Uses Plotly for dynamic charts (line plots, scatter plots, pie charts, bar charts) styled according to the theme.
*   **Thematic Notifications:** User feedback provided through Breaking Bad-themed messages (`st.success`, `st.info`, `st.error`).
*   **Error Handling:** Graceful error messages for common issues (e.g., invalid file upload, data fetching errors).
*   **(Bonus) Download Results:** Option to download predictions or cluster assignments as a CSV file ("Package the Batch").
*   **(Bonus) Feature Importance:** Visualizes feature importance for Linear Regression ("Chemical Composition Chart").
*   **(Bonus) Data Refresh:** Button to refresh Yahoo Finance data.

## Technology Stack 🛠️

*   **Language:** Python 3.8+
*   **Web Framework:** Streamlit
*   **Data Handling:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn
*   **Data Fetching:** yfinance
*   **Visualization:** Plotly, Matplotlib (for Confusion Matrix)
*   **Styling:** CSS
