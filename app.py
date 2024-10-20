import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load a free text-to-SQL model from Hugging Face
model_name = "tscholak/cxmefzzi"  # Pre-trained model for text-to-SQL
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

st.title('Natural Language to SQL Generator')
st.markdown("This tool generates SQL queries from natural language requests using a free Hugging Face model.")

# Input: Natural Language Request
natural_language_input = st.text_input('Describe your data request', 'Get all customers who signed up last month')

# Button to generate SQL
go = st.button('Generate SQL Query')

if go:
    try:
        # Tokenize and generate SQL query
        inputs = tokenizer.encode(f"translate to SQL: {natural_language_input}", return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=150)

        # Decode the generated query
        sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display the generated SQL query
        st.code(sql_query, language='sql')
        
    except Exception as e:
        st.exception(f"Exception: {e}")

st.markdown('___')
st.markdown('by [YourName]')
