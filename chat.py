import openai
import datetime
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import base64
import pandas as pd
import requests
from bs4 import BeautifulSoup


# Set your OpenAI API key
api_key = "OPENAI_API_KEY"  # Replace with your OpenAI API key
openai.api_key = api_key

model = "gpt-4"

# Hide 'Made with Streamlit' footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Create directories for logs
if not os.path.exists('Chat Logs'):
    os.makedirs('Chat Logs')
if not os.path.exists('Saved Chats'):
    os.makedirs('Saved Chats')

# Generate log file path
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join('Chat Logs', f'log_{timestamp}.txt')

# Initialize typed query history in session state
def scrape_google_scholar(query):
    # Construct the URL for Google Scholar search
    url = "https://scholar.google.com/scholar?q=" + "+".join(query.split())

    headers = {'User-Agent': 'Mozilla/5.0'}  # Header to mimic a browser request
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract relevant data: titles and URLs
        search_results = soup.find_all('div', class_='gs_ri')  # Google Scholar's result item container
        results = []
        for result in search_results:
            title = result.find('h3', class_='gs_rt').get_text()
            link = result.find('a', href=True)['href'] if result.find('a', href=True) else 'No link available'
            results.append({'title': title, 'link': link})

        return results
    else:
        return "Failed to retrieve data"

if 'typed_query_history' not in st.session_state:
    st.session_state.typed_query_history = []

def generate_bar_chart(df, column):
    fig = px.bar(df, x=column, title=f'Bar Chart of {column}')
    st.plotly_chart(fig)

def generate_histogram(df, column):
    fig = px.histogram(df, x=column, title=f'Histogram of {column}')
    st.plotly_chart(fig)

def show_advanced_stats(df):
    st.write("Skewness of each column:")
    st.write(df.skew())
    st.write("Kurtosis of each column:")
    st.write(df.kurtosis())

def build_linear_regression_model(df):
    st.sidebar.subheader("Linear Regression Model")

    # Filter DataFrame to numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) > 1:
        x_column = st.sidebar.selectbox("Select Feature Column", numeric_columns)
        y_column = st.sidebar.selectbox("Select Target Column", numeric_columns)
        X = df[[x_column]]
        y = df[y_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("Mean Squared Error:", mse)
        st.write("R^2 Score:", r2)
    else:
        st.sidebar.write("Not enough numeric columns for regression analysis.")

def generate_seaborn_plot(df):
    st.sidebar.subheader("Seaborn Plot Options")
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Pairplot", "Heatmap", "Boxplot"])
    
    if plot_type == "Pairplot":
        pairplot = sns.pairplot(df)
        fig = pairplot.fig
    elif plot_type == "Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax)
    elif plot_type == "Boxplot":
        selected_column = st.sidebar.selectbox("Choose Column for Boxplot", df.columns)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_column], ax=ax)

    st.pyplot(fig)

def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    st.title("Your AI Assistant for Data Analysis")
    st.sidebar.header("Settings")
    model_selection = st.sidebar.selectbox("Model Selection", ["gpt-4", "gpt-3.5"], index=0)
    temperature = st.sidebar.slider("Temperature Setting", 0.0, 1.0, 0.5)

    # Web Scraping Section
    st.sidebar.title("Web Scraping")
    query = st.sidebar.text_input("Enter your search query", "")
    if st.sidebar.button("Scrape"):
        scrape_results = scrape_google_scholar(query)
        st.write(scrape_results)

    # Data Upload
    st.sidebar.title("Data Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=["csv", "xlsx", "json", "parquet"])
    df = None

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]

        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_type == 'xlsx':
            df = pd.read_excel(uploaded_file)
        elif file_type == 'json':
            df = pd.read_json(uploaded_file)
        elif file_type == 'parquet':
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file format")

        if df is not None:
            st.write("Data Preview:", df.head())

            if st.sidebar.checkbox('Show Summary Statistics'):
                st.write(df.describe())

            if st.sidebar.checkbox('Show Advanced Statistics'):
                show_advanced_stats(df)

            viz_type = st.sidebar.selectbox("Choose Visualization Type", ["Bar Chart", "Histogram", "Scatter Plot", "Seaborn Plot"])
            if viz_type != "Scatter Plot":
                selected_column = st.sidebar.selectbox("Choose Column", df.columns)
                if viz_type == "Bar Chart":
                    generate_bar_chart(df, selected_column)
                elif viz_type == "Histogram":
                    generate_histogram(df, selected_column)
                elif viz_type == "Seaborn Plot":
                    generate_seaborn_plot(df)

            if st.sidebar.checkbox('Build Linear Regression Model'):
                build_linear_regression_model(df)

            if st.sidebar.button('Download Data as CSV'):
                download_csv(df)

    user_query = st.text_input("Enter your message or query", key="user_query")
    response_obj = None  # Initialize response_obj to None
    response = None      # Initialize response to None

    if user_query:
        response_obj = openai.ChatCompletion.create(
            model=model_selection,
            messages=[{"role": "user", "content": user_query}]
)
        response = response_obj.choices[0].message['content']

    # Write the response if it's available
    if response:
        st.session_state.typed_query_history.append({"user_query": user_query, "response": response})
        st.write(response)

        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"User: {user_query}\nAI: {response}\n\n")

    st.sidebar.title('Typed Query History')
    clear_typed_query_history = st.sidebar.button("Clear Typed Query History")

    if clear_typed_query_history:
        st.session_state.typed_query_history = []

    for i, entry in enumerate(st.session_state.typed_query_history):
        query = entry["user_query"]
        response = entry["response"]
        if st.sidebar.button(f"Query {i + 1}: {query}", key=f"typed_query_history_button_{i}"):
            st.write(f"Response for '{query}':")
            st.write(response)

if __name__ == "__main__":
    main()
