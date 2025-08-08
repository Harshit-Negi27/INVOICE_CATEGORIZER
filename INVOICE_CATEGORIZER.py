import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import streamlit as st
import easyocr
import tempfile
import os
from pdf2image import convert_from_path
from PIL import Image
import re

# Set the page configuration 
st.set_page_config(
    page_title="InvoSpell",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
    <style>
        h1 {
            font-size: 70px !important; 
            font-weight: bold;
            text-align: center;
            text-shadow: 4px 4px 8px #000000;
            margin-top: 20px;
            margin-bottom: 10px;
            
            /* --- Flashing, Colorful Heading --- */
            background-image: linear-gradient(45deg, #DA70D6, #FFD700, #8A2BE2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            color: transparent; /* Fallback for browsers that don't support the above */
            /* --------------------------------- */
        }
        .subtitle {
            font-size: 24px;
            color: #FFD700;
            text-align: center;
            font-style: italic;
            margin-bottom: 30px;
        }
        .subheader {
            font-size: 38px;
            font-weight: bold;
            color: #DA70D6;
            text-align: center;
        }
        .info-box {
            background-color: #2F4F4F;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stButton>button {
            color: white;
            background-color: #483D8B;
            border-radius: 20px;
            padding: 10px 20px;
            font-size: 20px;
        }
        .download-btn {
            background-color: #CD5C5C !important;
        }
        .dataframe-container {
            border: 2px solid #DA70D6;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the cleaned dataset 
df = pd.read_csv("cleaned_final_dataset.csv")
df.columns = ["text", "category"]
df = df.dropna(subset=["text", "category"])

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['category']
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# OCR Function
def run_ocr(file_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(file_path)
    
    headers_to_find = ["item description", "quantity", "unit price", "total"]
    header_boxes = []

    for (bbox, text, prob) in results:
        normalized_text = re.sub(r'[^a-zA-Z\s]', '', text.lower()).strip()
        if normalized_text in headers_to_find:
            header_boxes.append(bbox)

    if not header_boxes:
        return [res[1] for res in results if not re.search(r'total|subtotal|tax|invoice|bill|customer|address', res[1].lower())]

    x_min_table = min(box[0][0] for box in header_boxes)
    x_max_table = max(box[2][0] for box in header_boxes)
    y_max_headers = max(box[2][1] for box in header_boxes)
    
    item_lines = []
    
    for (bbox, text, prob) in results:
        x_min_line, y_min_line = bbox[0]
        x_max_line, y_max_line = bbox[2]
        
        if y_min_line > y_max_headers and x_min_line >= x_min_table - 20 and x_max_line <= x_max_table + 20 and prob > 0.5:
            cleaned_text = re.sub(r'[\d\$,₹%.]+', '', text).strip()
            if "for your business" in cleaned_text.lower():
                continue
            
            if len(cleaned_text.split()) > 1 and len(cleaned_text) > 5:
                item_lines.append(cleaned_text)
                
    return item_lines

# --- UI Layout ---
st.title("InvoSpell 🪄")
st.markdown("<p class='subtitle'>Cast a spell, get your invoice sorted!</p>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div class="info-box">
        <p style="font-size: 20px;">
            <b>Welcome to the future of expense tracking!</b> 🚀
            This tool uses powerful AI to instantly digitize and categorize your invoices with dazzling speed and accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<p class='subheader'>Upload Your Magic Invoice ✨</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG, PDF",
        type=["jpg", "jpeg", "png", "pdf"]
    )
    
    if uploaded_file:
        st.success("🎉 File uploaded successfully! Let the magic begin...")
    else:
        st.info("Awaiting file upload...")

with col2:
    st.markdown("<p class='subheader'>🔮 Your Categorized Results</p>", unsafe_allow_html=True)
    st.markdown("---")

    if uploaded_file:
        with st.spinner("🧙‍♂️ Brewing a magical potion... Running OCR and AI Model..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            if uploaded_file.name.lower().endswith(".pdf"):
                try:
                    pages = convert_from_path(temp_path)
                    image_path = temp_path + ".jpg"
                    pages[0].save(image_path, 'JPEG')
                    os.remove(temp_path)
                except (IOError, IndexError) as e:
                    st.error(f"Error processing PDF: {e}")
                    st.stop()
            else:
                image_path = temp_path

            ocr_lines = run_ocr(image_path)
            os.remove(image_path)

        if len(ocr_lines) == 0:
            st.warning("No text found in invoice.")
        else:
            st.success("✅ Poof! Text extracted and categorized successfully!")
            
            if len(ocr_lines) > 0:
                X_test = vectorizer.transform(ocr_lines)
                preds = model.predict(X_test)
                
                df_result = pd.DataFrame({"Line Item": ocr_lines, "Predicted Category": preds})
                
                st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
                st.dataframe(df_result, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name='categorized_invoice.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            else:
                st.warning("Could not identify any relevant line items to categorize.")
    else:
        st.write("Upload an invoice on the left to see the categorized results here.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #808080;'>🔮 Powered by AI Magic & Streamlit ✨</p>", unsafe_allow_html=True)