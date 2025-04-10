import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import torch.nn.functional as F

st.set_page_config(page_title="Smart QA Chatbot", layout="centered")

@st.cache_resource
def load_model():
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

def get_answer(context, question, tokenizer, model, threshold=0.2):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)

    start_idx = torch.argmax(start_probs)
    end_idx = torch.argmax(end_probs) + 1
    confidence = (start_probs[0][start_idx] * end_probs[0][end_idx - 1]).item()

    if confidence < threshold:
        return "🤔 I couldn't find a confident answer in the given context.", confidence

    answer_tokens = input_ids[start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer.strip(), confidence

def main():
    st.title("💬 Smart QA Chatbot")
    st.markdown("Ask questions based on the context you provide below!")

    tokenizer, model = load_model()

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.expander("📄 Example Context"):
        st.markdown("You can use this for testing:")
        st.code("""The Moon is Earth's only natural satellite. It is about one-quarter the diameter of Earth and is the fifth largest satellite in the Solar System. The Moon orbits Earth at an average distance of 384,400 km.""", language="markdown")

    context = st.text_area("📝 Provide Context", height=150)

    if context:
        question = st.chat_input("Ask a question based on the context...")

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer, confidence = get_answer(context, question, tokenizer, model)
                        st.markdown(f"**Answer:** {answer}")
                        st.markdown(f"**Confidence:** {confidence:.2f}")
                        st.session_state.history.append((question, answer, confidence))
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    if st.session_state.history:
        st.divider()
        st.subheader("🕘 Past Q&A")
        for q, a, c in reversed(st.session_state.history[-5:]):
            st.markdown(f"**Q:** {q}\n\n**A:** {a} *(Confidence: {c:.2f})*")

if __name__ == "__main__":
    main()
