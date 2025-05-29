import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Load dữ liệu mẫu
@st.cache_data
def load_data():
    return pd.read_csv("sample_reviews.csv")  # bạn tự tạo file CSV gồm cột "review"

# Hàm sentiment analysis
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit layout
st.set_page_config(layout="wide")
st.title("📊 Phân tích đánh giá địa điểm - KHÁCH HÀNG LÀ THƯỢNG ĐẾ")

data = load_data()
st.write("## Dữ liệu review mẫu", data.head())

# Phân tích sentiment
with st.spinner("🔍 Đang phân tích sentiment..."):
    data['Sentiment'] = data['review'].apply(analyze_sentiment)
    st.success("✅ Hoàn tất phân tích sentiment!")
    st.write(data[['review', 'Sentiment']])

# Biểu đồ sentiment
st.write("## 📈 Phân bố cảm xúc")
st.bar_chart(data['Sentiment'].value_counts())

# Topic modeling với BERTopic
with st.spinner("🧠 Đang thực hiện topic modeling..."):
    model = BERTopic(embedding_model=SentenceTransformer("all-MiniLM-L6-v2"))
    topics, _ = model.fit_transform(data['review'].tolist())
    st.success("✅ Topic modeling hoàn tất!")

# Hiển thị kết quả topic
st.write("## 📌 Các chủ đề nổi bật trong review")
topic_info = model.get_topic_info()
st.dataframe(topic_info)

selected_topic = st.selectbox("Chọn một topic để xem chi tiết:", topic_info['Name'][1:])
if selected_topic:
    topic_id = topic_info[topic_info['Name'] == selected_topic]['Topic'].values[0]
    st.write(model.get_topic(topic_id))

