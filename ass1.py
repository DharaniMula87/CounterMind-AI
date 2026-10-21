import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


st.set_page_config(
    page_title="CounterMind AI",
    layout="centered"
)

st.title("CounterMind")
st.subheader("Challenge My Thinking")
st.write(
    "An AI thinking partner that questions your assumptions, "
    "challenges your arguments, and helps you think critically."
)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_tokens=1200,
    api_key=GROQ_API_KEY
)


MODES = {
    "Analyze Argument": """
Analyze the user's argument critically.

Identify:
1. Main claim
2. Weak assumptions
3. Logical gaps
4. Counterarguments
5. Missing evidence
6. Suggestions for improvement

Be respectful and explain your reasoning.
""",

    "Find Missing Evidence": """
Examine the user's argument and identify claims
that need evidence.

Separate facts, opinions, and assumptions.
Suggest what data, research, or reliable sources
would help verify each important claim.

Do not invent facts or pretend you verified sources.
""",

    "Strengthen My Argument": """
Improve the user's argument while preserving
their intended position.

Make it clearer, more logical, balanced,
and precise.

Explain the major weaknesses in the original
and provide a stronger revised argument.
Do not invent supporting evidence.
""",

    "Devil's Advocate": """
Act as a thoughtful devil's advocate.

Argue the opposite side of the user's position
as persuasively as possible.

Include:
1. The strongest opposing argument
2. Weaknesses in the user's reasoning
3. Alternative perspectives
4. Questions the user should consider

Be intellectually honest. Do not invent facts.
"""
}


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are CounterMind, a critical-thinking assistant.

Your goal is not to agree automatically with the user.
Help the user evaluate ideas through logical reasoning.

Rules:
- Be respectful, direct, and constructive.
- Distinguish facts from opinions and assumptions.
- Explain why an argument may be weak.
- Consider alternative perspectives.
- Never invent evidence, statistics, or citations.
- Admit uncertainty when information is insufficient.
- Organize your response with headings and bullet points.

Selected analysis instructions:
{instructions}
"""
    ),
    (
        "human",
        "Analyze this argument:\n\n{argument}"
    )
])

chain = prompt | llm | StrOutputParser()



argument = st.text_area(
    "Enter your argument or opinion",
    placeholder=(
        "Example: Online education is better than "
        "classroom education because students can learn from home."
    ),
    height=150
)

selected_mode = st.selectbox(
    "Choose an analysis mode",
    [
        "Analyze Argument",
        "Find Missing Evidence",
        "Strengthen My Argument",
        "Devil's Advocate"
    ]
)



def analyze_argument(user_argument, mode):

    return chain.invoke({
        "argument": user_argument,
        "instructions": MODES[mode]
    })


col1, col2 = st.columns(2)

with col1:
    analyze_clicked = st.button(
        "Analyze Argument",
        use_container_width=True
    )

with col2:
    devil_clicked = st.button(
        " Devil's Advocate",
        use_container_width=True
    )


if analyze_clicked or devil_clicked:

    if not argument.strip():
        st.warning("Please enter an argument first.")

    else:
    
        mode = (
            "Devil's Advocate"
            if devil_clicked
            else selected_mode
        )

        try:
            with st.spinner("CounterMind is challenging your thinking..."):

                response = analyze_argument(argument, mode)

            st.divider()
            st.subheader(f" {mode}")
            st.markdown(response)

        except Exception as e:
          st.error("CounterMind Error")
          st.write("Error type:", type(e).__name__)
          st.code(str(e))


st.divider()
st.caption(
    "CounterMind AI | Think deeper. Question assumptions. "
    "Build stronger arguments."
)