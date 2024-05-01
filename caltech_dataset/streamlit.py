import streamlit as st
import tensorflow as tf
from caltech_test import test_model  
from caltech_train import train_model  


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('Model_320')  # Update path if necessary
    return model

model = load_model()

st.title('Video Frame Prediction Tool')

# Interface for Model Training
if st.button('Train Model'):
    train_model()
    st.success('Model trained successfully!')

# Interface for Model Testing
st.header('Test the Model')
# You might want to allow users to upload a video file or provide inputs for testing
uploaded_file = st.file_uploader("Choose a file...")
if uploaded_file is not None:
    # Assume test_model function exists and is ready to handle file input directly
    result = test_model(uploaded_file, model)
    st.write('Model output:', result)

# Display model summary
st.header('Model Summary')
model.summary(print_fn=lambda x: st.text(x))

# Run this app with `streamlit run your_script.py`
