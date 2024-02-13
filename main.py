import os
import sys
from dotenv import load_dotenv, find_dotenv

import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

from tools import getKeyword, SearchPaper, SearchAndDownloadPaper, SavePaper, DownloadVectorStore

sys.path.append('../..')
_ = load_dotenv(find_dotenv("en.env"))
openai.api_key = os.environ['OPENAI_API_KEY']


llm = ChatOpenAI(temperature=0.5)

translate_English_template = """
Translate the following query into English.
{query}
"""

translate_Chinese_template = """
Translate the following query into Chinese.
{query}
"""


searchPaper_template = """
You are a AI technology blogger who write for shortly summarizing papers on Arxiv. You write summary based on a list of text with content of Title, Abstract, URL: 
##start\n {text} \n##end 
For each paper you write the Title, a Chinese quick summary including key information from the Title and Abstract after Title, then, attach the URL after summary. 
Summaries:
"""

structure_template = """
Use the following context to briefly answer the question in Chinese at the end.
give an ordered list of your answer.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}

Question:关于开发大语言模型支持的科研论文写作助手的研究论文写作的大纲是怎样的?
Answer:
对于关于开发大语言模型支持的科研论文写作助手的科研论文，其写作大纲可以为如下形式：

**1. 引言：**

    介绍研究背景和大语言模型在科研写作中的潜在应用。

**2. 相关工作：**

    回顾大语言模型在科研写作和自然语言处理领域的相关研究。

**3. 方法论：**

    描述使用的大语言模型。

    说明开发科研论文写作助手的步骤和方法。

**4. 实验设计与结果：**

    定义实验目标和评价标准。

    展示模型的测试结果和性能分析。

**5. 应用案例：**

    提供实际案例，展示工具的实用性。

**6. 结论和未来工作：**

    总结研究成果，讨论未来的改进方向。

Question:{question}
Answer:
"""


trans_En_prompt = PromptTemplate(template=translate_English_template, input_variables=["query"])
trans_Ch_prompt = PromptTemplate(template=translate_English_template, input_variables=["query"])
paper_prompt = PromptTemplate(template=searchPaper_template, input_variables=["text"])
qa_prompt = PromptTemplate.from_template(template=structure_template)

llm_trans_En_chain = LLMChain(
    llm=llm,
    prompt=trans_En_prompt,
    verbose=True,
    output_key="topic"
)

llm_trans_Ch_chain = LLMChain(
    llm=llm,
    prompt=trans_Ch_prompt,
    verbose=True,
    output_key="answer"
)


def main():
    llm = ChatOpenAI(temperature=0)
    st.set_page_config(
        page_title="AI英语科研论文写作助手",
    )
    st.header("AI英语科研论文写作助手")
    # 初始化聊天历史
    st.chat_message("assistant").write("您本次科研论文写作的主题是什么呢？")
    # container = st.container()
    if topic := st.chat_input():
        st.chat_message("user").write(topic)

        # # 加载相关论文存入向量数据库
        # keywords = getKeyword(topic)
        # path = "docs/related_works"
        # SearchAndDownloadPaper(keywords, path)
        # SavePaper(path)

        # # 搜索主题相关论文推荐给用户
        # summaries = SearchPaper(keywords)
        # with st.chat_message("assistant"):
        #     st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        #     llm_paper_chain = LLMChain(
        #         llm=llm,
        #         prompt=paper_prompt,
        #         verbose=True,
        #         callbacks=[st_cb]
        #     )
        #     papers = llm_paper_chain.run(summaries)

        with st.chat_message("assistant"):
            #输出文章结构
            st_cb = StreamlitCallbackHandler(st.container())
            from langchain.prompts import PromptTemplate
            from langchain.chains import RetrievalQA
            path = "docs/chroma"
            vectordb = DownloadVectorStore(path)
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectordb.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_prompt},
            )
            seqChain = SequentialChain(chains=[llm_trans_En_chain, qa_chain],
                                       input_variables=["query"],
                                       verbose=True,
                                       callbacks=[st_cb]
                                       )
            result = seqChain({
                "query": "关于" + topic + "的研究论文写作的大纲是怎样的?"
            })
            st.write(result['result']+"\n\n下面请由你来写作，我们来润色吧！\n\n ⚠️ 请按照以下格式写作某一章节并逐一提交：\n\n[章节标题]\n\n[章节内容]")

    # 显示聊天历史
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"系统消息：{message.content}")
    # with container:
    #     with st.form(key='my_form', clear_on_submit=True):
    #         user_input = st.text_area(label='Message: ', key='input', height=100)
    #         submit_button = st.form_submit_button(label='Send')
    # if submit_button and user_input:
    #     st.session_state.messages.append(HumanMessage(content=user_input))
    #     with st.spinner("正在输入..."):
    #         response = llm(st.session_state.messages)
    #     st.session_state.messages.append(AIMessage(content=response.content))

if __name__ == '__main__':
    main()