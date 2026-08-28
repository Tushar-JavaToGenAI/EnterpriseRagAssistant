import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage

class GroundedAnswerGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0
        )

    def contextualize_query(self, query: str, chat_history: List[BaseMessage]) -> str:
        """Converts multi-turn questions into standalone searchable queries."""
        if not chat_history:
            return query

        condense_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference previous context, formulate a standalone question. Do NOT answer the question, just reformulate it."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        chain = condense_prompt | self.llm
        reformulated = chain.invoke({"chat_history": chat_history, "question": query})
        return reformulated.content

    def generate_answer(self, query: str, retrieved_docs: List[Document], chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """Generates grounded answer strictly relying on retrieved context."""
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))
        context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        system_prompt = (
            "You are an enterprise HR & Policy Assistant. Answer the user question using ONLY the provided context.\n"
            "Strict Constraints:\n"
            "1. If the exact answer is not present in the context, explicitly state: 'The provided documents do not contain information regarding this request.'\n"
            "2. Never assume, fabricate, or extrapolate policies outside the given context.\n"
            "3. Ground all answers directly on the retrieved text.\n\n"
            f"Context:\n{context_str}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        chain = qa_prompt | self.llm
        response = chain.invoke({"chat_history": chat_history, "question": query})

        return {
            "answer": response.content,
            "sources": sources,
            "retrieved_context": [d.page_content for d in retrieved_docs]
        }