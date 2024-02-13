import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from pprint import pprint

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from tools import search_arxiv, paper_print_text, search_for_outline, search_for_info, print_outline, outline_to_dict, \
    save_text, split_pdf_by_outline, print_section_outline

sys.path.append('../..')
_ = load_dotenv(find_dotenv("en.env"))

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k-0613",
    temperature=0
)


def generate_outline(field, topic, num):
    """生成论文大纲"""

    papers = search_for_outline(field, topic, num)

    outline_template = """
Generate the outline of the following {field} English scientific research paper based on the title and abstract.
The outline should include like "Introduction", "Related Work", "Methodology", "Experiment and Evaluation", "Discussion" and "Conclusion"
Please output in a dictionary format, structured as {{"title": "<Paper Title>", "outline": {{}}}}.
Within the "outline" dictionary, each key should be a chapter title, and each chapter title's corresponding value should be a python list of key points for that chapter.
    """

    research_template = """
Here are example titles and outlines for the {field} English scientific research paper on similar topics:
{example}
Based on the example, generate a title and outline for a {field} English scientific research paper on '{topic}'.
Please output in a dictionary format, structured as {{"title": "<Paper Title>", "outline": {{}}}}.
Within the "outline" dictionary, each key should be a chapter title, and each chapter title's corresponding value should be a python list of key points for that chapter.

"""

    out_prompt = PromptTemplate(
        template=outline_template,
        input_variables=['field', 'papers']
    )

    res_prompt = PromptTemplate(
        template=research_template,
        input_variables=['field', 'example', 'topic']
    )

    llm_outline_chain = LLMChain(
        llm=llm,
        prompt=out_prompt,
        verbose=True,
        output_key="example"
    )
    llm_research_chain = LLMChain(
        llm=llm,
        prompt=res_prompt,
        verbose=True,
        output_key="outline"
    )
    seqChain = SequentialChain(chains=[llm_outline_chain, llm_research_chain],
                               input_variables=["field", "topic", "papers"],
                               verbose=True,
                               )
    response = seqChain({
        "field": field,
        "topic": topic,
        "papers": papers
    })
    return eval(response['outline'])


def string_to_markdown(text):
    markdown_template = """
    Convert the following text into markdown format.
    The heading level of title of the paper should be 2. The heading level of title of sections should be 3.
    {text}
    """
    mark_prompt = PromptTemplate(
        template=markdown_template,
        input_variables=["text"],
    )
    llm_markdown_chain = LLMChain(
        llm=llm,
        prompt=mark_prompt,
        verbose=True,
        output_key="markdown"
    )
    response = llm_markdown_chain.run({"text": text})
    return response


def generate_citation(papers, papers_dict):
    cite_template = """
Here are example papers and citations:
Arxiv: http://arxiv.org/abs/1502.05041v1
Title: AMAS: optimizing the partition and filtration of adaptive seeds to speed up read mapping
Authors: [arxiv.Experiment and Result.Author('Ngoc Hieu Tran'), arxiv.Experiment and Result.Author('Xin Chen')]
Submitted Date: 2015-02-18 02:59:08+00:00
Comment: IEEE/ACM Transactions on Computational Biology and Bioinformatics,
2016
Journal: None
Citation: Ngoc Hieu Tran and Xin Chen. 2015. AMAS: optimizing the partition and filtration of adaptive seeds to speed up read mapping. IEEE/ACM Transactions on Computational Biology and Bioinformatics (2016).

Arxiv: http://arxiv.org/abs/1111.5149v1
Title: Lock-in detection for pulsed electrically detected magnetic resonance
Authors: [arxiv.Experiment and Result.Author('Felix Hoehne'), arxiv.Experiment and Result.Author('Lukas Dreher'), arxiv.Experiment and Result.Author('Jan Behrends'), arxiv.Experiment and Result.Author('Matthias Fehr'), arxiv.Experiment and Result.Author('Hans Huebl'), arxiv.Experiment and Result.Author('Klaus Lips'), arxiv.Experiment and Result.Author('Alexander Schnegg'), arxiv.Experiment and Result.Author('Max Suckert'), arxiv.Experiment and Result.Author('Martin Stutzmann'), arxiv.Experiment and Result.Author('Martin S. Brandt')]
Submitted Date: 2011-11-22 10:40:07+00:00
Comment: 4 pages, 2 figures
Journal: None
Citation: Felix Hoehne, Lukas Dreher, Jan Behrends, Matthias Fehr, Hans Huebl, Klaus Lips, Alexander Schnegg, Max Suckert, Martin Stutzmann and Martin S. Brandt. 2011. Lock-in detection for pulsed electrically detected magnetic resonance. Arxiv Preprint Arxiv: 1111.5149 (2011)

Based on the example, generate the citations for each of following papers.
Please output in a Python list format.
{papers}

Answer:
    """

    cite_prompt = PromptTemplate(
        template=cite_template,
        input_variables=["papers"]
    )

    llm_cite_chain = LLMChain(
        llm=llm,
        prompt=cite_prompt,
    )
    if len(papers_dict) > 0:
        papers = llm_cite_chain.run(papers)
        papers = eval(papers)
        if len(papers) == len(papers_dict):
            for i in range(len(papers)):
                papers_dict[i]['Citation'] = papers[i]
    return papers_dict


def save_pdf_chroma(path):
    from langchain.document_loaders import PyPDFLoader
    loaders = []
    pdf_list = os.listdir(path)
    pdf_list = [pdf for pdf in pdf_list if pdf != 'chroma']
    for pdf in pdf_list:
        loaders.append(PyPDFLoader(os.path.join(path, pdf)))
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # 按句子分割
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # 切割文档
    splits = text_splitter.split_documents(docs)
    embedding = OpenAIEmbeddings()
    path = os.path.join(path, 'chroma')
    os.makedirs(path, exist_ok=True)
    # 创建向量数据库
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=path
    )
    print(vectordb._collection.count())


def save_text_chroma(papers, path):
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    loader = TextLoader(papers)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        separator='\n\n'
    )
    splits = text_splitter.split_documents(documents)
    if len(splits) > 0:
        save_path = os.path.join(path, 'chroma')
        embedding = OpenAIEmbeddings()
        os.makedirs(save_path, exist_ok=True)
        # 创建向量数据库
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=save_path
        )


def revise_text(field, topic, section, outlines, draft, path):
    points = outlines['outline'][section]['points']
    outline = print_outline(outlines)
    vec_path = os.path.join(path, 'chroma')
    context = {
        'Content': '',
        'References': ''
    }
    for point in points:
        papers_dict = search_arxiv(field, point, 5)
        papers = paper_print_text(papers_dict, ['Abstract'])
        papers = paper_print_text(generate_citation(papers, papers_dict),
                                  ['Arxiv', 'Authors', 'Submitted Date', 'Comment', 'Journal'])
        papers = save_text(papers, point, path)
        save_text_chroma(papers, path)
        embedding = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory=vec_path, embedding_function=embedding)

        question = "What is " + point + "?"

        docs = vectordb.max_marginal_relevance_search(question)
        information = "\n\n".join([doc.page_content for doc in docs])

        para_template = """
You are a helpful English scientific research paper writing assistant.
You are trying to write the {section} section of the {field} English scientific research paper about '{topic}'.
You will be given the related papers about '{point}', the outline of the paper and the draft of the {section} section.
Please continue to write one paragraph to discuss about '{point}'.
Cite the used related papers in APA format at corresponding places and add in References list in order.
Focus on the contextual coherence and avoid writing redundant content with other key points.
Keep paragraph length between 50-150 words.
Please output in a dictionary format, structured as {{"Paragraph": "<Generated Paragraph>", "References": "<References>"}}.

# RELATED PAPERS
{information}

# OUTLINE
{outline}

# DRAFT
{draft}

#FINISHED TEXT
{text}

# ANSWER
"""

        para_prompt = PromptTemplate(
            template=para_template,
            input_variables=["field", "topic", "section", "outline", "point", "draft", "information", "text"],
        )
        llm_para_chain = LLMChain(
            llm=llm,
            prompt=para_prompt,
            verbose=True,
            output_key="context"
        )
        result = llm_para_chain.run({
            "field": field,
            "topic": topic,
            "section": section,
            "outline": outline,
            "point": point,
            "draft": draft,
            "information": information,
            "text": context['Content'] + '\n\nReferences:\n' + context['References']
        })
        result = eval(result)
        context['Content'] += '\n'
        context['Content'] += result['Paragraph']
        context['References'] = result['References']
    print(context['Content'] + '\n\nReferences:\n' + context['References'])
    return context


def search_google_scholar(topic):
    from langchain.tools.google_scholar import GoogleScholarQueryRun
    from langchain.utilities.google_scholar import GoogleScholarAPIWrapper

    tool = GoogleScholarAPIWrapper()
    return tool.run(topic)


def generate_outline_test(field, topic, path):
    from langchain.document_loaders import PyPDFLoader
    from langchain.chains.summarize import load_summarize_chain

    loaders = []
    pdf_list = os.listdir(path)
    pdf_list = [pdf for pdf in pdf_list if pdf != 'chroma']

    for pdf in pdf_list:
        loaders.append(PyPDFLoader(os.path.join(path, pdf)))

    related_outline_template = """
You are an English Scientific Research Paper Writing Assistant for Extracting the outline of {field} English scientific research paper.
Generate a concise writing outline, including Introduction, Related Work, Method, Experiment and Experiment and Result, Discussion and Conclusion, of the following summary of {field} English scientific research paper.
Please output in a dictionary format, structured as {{"title": "<Paper Title>", "outline": {{}}}}.
Within the "outline" dictionary, each key should be a chapter title, and each chapter title's corresponding value should be a python list of key points for that chapter.

{summary}
    """

    rela_prompt = PromptTemplate(
        template=related_outline_template,
        input_variables=['field', 'summary'],
    )

    llm_summarize_chain = load_summarize_chain(
        llm,
        chain_type='map_reduce',
        output_key='summary'
    )

    llm_related_chain = LLMChain(
        llm=llm,
        prompt=rela_prompt,
        verbose=True,
    )
    outline = ''
    num = len(loaders)
    if num > 0:
        for loader in loaders:
            docs = loader.load_and_split()
            summary = llm_summarize_chain.run(docs)
            result = llm_related_chain.run({
                'field': field,
                'summary': summary
            })
            outline += result
            outline += '\n\n'

        outline_template = """
You are an English Scientific Research Paper Writing Assistant for Planning the outline of {field} English scientific research paper.
Here are {num} example titles and outlines for the related works about '{topic}':

{outline}

Referring to the example, generate a title and a brief outline, including Introduction, Related Work, Method, Experiment and Experiment and Result, Discussion, Conclusion, for a {field} English scientific research paper on '{topic}'.
Please output in a dictionary format, structured as {{"title": "<Paper Title>", "outline": {{}}}}.
Within the "outline" dictionary, each key should be a chapter title, and each chapter title's corresponding value should be a python list of key points for that chapter.
    """

        out_prompt = PromptTemplate(
            template=outline_template,
            input_variables=['field', 'topic', 'outline', 'num']
        )

        llm_outline_chain = LLMChain(
            llm=llm,
            prompt=out_prompt,
            verbose=True,
        )

        return eval(llm_outline_chain.run({
            'field': field,
            'topic': topic,
            'outline': outline,
            'num': num
        }))
    else:
        outline_template = """
You are an English Scientific Research Paper Writing Assistant for Planning the outline of {field} English scientific research paper.
Generate a title and a brief outline for a {field} English scientific research paper on '{topic}'.
Please output in a dictionary format, structured as {{"title": "<Paper Title>", "outline": {{}}}}.
Within the "outline" dictionary, each key should be a chapter title, and each chapter title's corresponding value should be a python list of key points for that chapter.
        """

        out_prompt = PromptTemplate(
            template=outline_template,
            input_variables=['field', 'topic']
        )

        llm_outline_chain = LLMChain(
            llm=llm,
            prompt=out_prompt,
            verbose=True,
        )

        return eval(llm_outline_chain.run({
            'field': field,
            'topic': topic,
        }))


def revise_text_test(field, topic, section, outlines, draft, ref_path, vec_path):
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.document_loaders import PyPDFLoader
    from langchain.chains.summarize import load_summarize_chain

    print(draft)

    points = outlines['outline'][section]['points']
    outline = print_outline(outlines)

    # 加载参考文献并总结主要内容
    references = ''
    loaders = []
    pdf_list = os.listdir(ref_path)
    pdf_list = [pdf for pdf in pdf_list if pdf != 'chroma']
    for pdf in pdf_list:
        loaders.append(PyPDFLoader(os.path.join(ref_path, pdf)))

    llm_summarize_chain = load_summarize_chain(
        llm,
        chain_type='map_reduce',
        output_key='summary'
    )

    num = len(loaders)
    if num > 0:
        for loader in loaders:
            pdfname = loader.file_path.split('/')
            pdfname = pdfname[len(pdfname) - 1]
            docs = loader.load_and_split()
            summary = llm_summarize_chain.run(docs)
            references = references + 'file name: ' + pdfname + '\n' + summary + '\n\n'

    # 找出相关文章对应的章节内容
    embedding = OpenAIEmbeddings()

    vectordb = Chroma(persist_directory=vec_path, embedding_function=embedding)
    print(vectordb._collection.count())

    ques_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
You are an AI language model assistant. 
Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {question}
""",
    )

    llm_chain = LLMChain(llm=llm, prompt=ques_prompt, verbose=True)

    retriever = MultiQueryRetriever(
        retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
    )  # "lines" is the key (attribute name) of the parsed output

    related_works = retriever.get_relevant_documents(
        query="What is the main content in " + section + "?"
    )

    related_works = "\n\n".join([doc.page_content for doc in related_works])

    # 按照要点一段一段续写一个章节
    text = ''
    for i in range(len(points)):
        para_template = """
You are a helpful English scientific research paper writing assistant.
You are trying to write the {section} section of the {field} English scientific research paper about '{topic}'.
You will be given the {section} section of the related papers, the references, the outline of the paper, the draft of the {section} section and the finished text of {section} in this paper.
Please refer to the writing style of the related paper to continue to write one paragraph to discuss about '{point}'.
Cite the used filename of related papers at corresponding places.
Focus on the contextual coherence and avoid writing redundant content with other key points.
Keep paragraph length between 50-150 words.

# OUTLINE
{outline}

# RELATED PAPERS
{related_works}

# REFERENCES
{references}

# DRAFT
{draft}

#FINISHED TEXT
{text}

# ANSWER
"""

        para_prompt = PromptTemplate(
            template=para_template,
            input_variables=["field", "topic", "section", "outline", "point", "draft", "related_works", "references",
                             "text"],
        )
        llm_para_chain = LLMChain(
            llm=llm,
            prompt=para_prompt,
            verbose=True,
        )
        result = llm_para_chain.run({
            "field": field,
            "topic": topic,
            "section": section,
            "outline": outline,
            "point": points[i],
            "draft": draft[i],
            "related_works": related_works,
            "references": references,
            "text": text
        })
        text += '\n\n'
        text += result
    return text


def get_section_points(section, content):
    points_template = """
You are a helpful English scientific research paper writing assistant for summarizing the main content of the content in the paper.
Here is an example. The input is the content of the Introduction section of an English scientific research paper and the output is the main content in this section.

example:
# INPUT
Unknown words, which we defined as words that the user spends extra time thinking about, can significantly increase the reading difficulty for English as a Second Language (ESL) learners [19, 26], since vocabulary knowledge is considered an essential feature of reading ability. Automatic detection of unfamiliar words can help users improve reading fluency and reading experience [14]. Previ- ous works used eye movement features such as fixation duration and the number of regressions to detect unknown words, consid- ering the strong correlation between eye movement and reading difficulty [7, 9, 18]. However, all these methods are based on dedi- cated eye-tracking devices. Even for portable eye trackers, prices of hundreds or thousands of dollars prevent most people from using these technologies in their daily lives.
The ubiquity of webcams makes it possible to track eye move- ment non-obtrusively. Researchers have used webcam-based eye- tracking methods to analyze users’ reading behavior [13], but the low precision [21] makes it infeasible to extract eye movement features proposed by previous works. Therefore, these reading anal- ysis approaches cannot be directly transferred to webcam-based unknown word detection, which requires fine-grained tracking.
In this work, we detect unknown words using the webcam for ESL learners by integrating gaze and text information with the help of transformer-based language models. Our method tracks eye movement using WebGazer [22] and embeds the positional information of gaze and texts with Long Short-term Memory [10] (LSTM) models. In order to improve the model’s performance given only noisy gaze data acquired by webcam, we leverage pre-trained language models to encode the context information for assistance. The gaze and context data are then fed to the model’s classifier to predict the targeted unknown word. The F1-score of our model is 75.73% which is higher than the text- and gaze-only model. We also conduct a user study to discover the needs of ESL learners and explore the design scope according to our method. The results show that our future design should focus more on proper nouns, multi-meaning words, and long and complex sentences, as well as exploring ways to reduce disturbance to the user’s reading process through interactive design. In summary, the main contributions of this work include the following:
1) We propose GazeReader, a novel webcam-based unknown word detecting method that leverages gaze and text information for ESL learners. The accuracy is 98.09%, the F1-score is 75.73%, and the cross-user F1-score is 78.26%.
2) We explore the design scope for ESL reading based on our method, guiding our future design direction towards proper nouns, multi-meaning words, and long and complex sentences.

#OUTPUT
['Background on ESL Learners' Challenges', 'Importance of Detecting Unknown Words', 'Existing Methods and Their Limitations', 'Overview of GazeReader Solution']

Referring to the example, summarize the main content of the input {section} section of the English scientific research paper.
The number of main content should be less than 5.
Please output in a python list format, each main content of this section should be one of the element in the list.

#INPUT
{content}
"""
    point_prompt = PromptTemplate(
        template=points_template,
        input_variables=['content', 'section']
    )
    llm_point_chain = LLMChain(
        llm=llm,
        prompt=point_prompt,
    )
    return eval(llm_point_chain.run({
        'section': section,
        'content': content
    }))


def get_related_work_outline(paper):
    title = paper[0]
    sections = classify_section(paper)
    chapters = paper[1]
    content = {}
    pprint(sections)
    for section in sections:
        chapter_class = sections[section]
        text = ''
        for chapter in chapter_class:
            for ch in chapters:
                if chapter in ch:
                    chapter = ch
                    break
            text += paper[2][chapter]
            text += '\n'
        content[section] = text
    sections = [section for section in sections]
    outline = title + "\n"
    for section in sections:
        if content[section]:
            outline += section
            outline += '\n'
            points = get_section_points(section, content[section])
            for point in points:
                outline += '- '
                outline += point
                outline += '\n'
        outline += '\n\n'
    return outline, (title, sections, content)


def classify_section(work):
    summary_template = """
You are an English Scientific Research Paper Writing Assistant.
You will be given the content of {section} section in an English scientific research paper.
Summarize the section in less than 100 words.
# CONTENT
{content}
# SUMMARY
"""
    section_template = """
You are an English Scientific Research Paper Writing Assistant.
You will be given the title of an English Scientific Research Paper, chapter titles, the summaries of all chapters in the paper.
Classify the chapters into 5 sections: Introduction, Related Work, Method, Experiment and Experiment and Result, Discussion.
Please output in a dictionary format, structured as {{"Introduction": [], "Related Work": [], "Method": [], "Experiment and Experiment and Result": [], "Discussion": []}}
Each section's corresponding value is a not-empty python list of the chapters that is classified into this section.
{paper}
"""

    sum_prompt = PromptTemplate(
        template=summary_template,
        input_variables=['content', 'section']
    )

    sec_prompt = PromptTemplate(
        template=section_template,
        input_variables=['paper']
    )

    llm_sum_chain = LLMChain(
        llm=llm,
        prompt=sum_prompt,
    )

    llm_class_chain = LLMChain(
        llm=llm,
        prompt=sec_prompt,
        verbose=True,
    )

    context = work[0] + '\n'
    sections = work[2]
    for section in sections:
        context += section
        context += '\n'
        summary = llm_sum_chain.run({
            'section': section,
            'content': sections[section]
        })
        context += summary
        context += '\n\n'
    result = llm_class_chain.run({
        'paper': context
    })
    return eval(result)


def generate_outline_test1(field, topic, outline):
    if len(outline) > 0:
        outline_template = """
You are an English Scientific Research Paper Writing Assistant for Planning the outline of {field} English scientific research paper.
Here are some examples for the related works about '{topic}', each example includes the title and the outline:

{outline}
Referring to the example, generate a title and a brief outline, including Introduction, Related Work, Method, Experiment and Experiment and Result, Discussion, for a {field} English scientific research paper on '{topic}'.
Please output in a dictionary format, structured as {{"title": "<Paper Title>", "outline": {{}}}}.
Within the "outline" dictionary, each key should be a chapter title, and each chapter title's corresponding value should be a python list of key points for that chapter.
"""

        out_prompt = PromptTemplate(
            template=outline_template,
            input_variables=['field', 'topic', 'outline', 'num']
        )

        llm_outline_chain = LLMChain(
            llm=llm,
            prompt=out_prompt,
        )

        return eval(llm_outline_chain.run({
            'field': field,
            'topic': topic,
            'outline': outline,
        }))
    else:
        outline_template = """
You are an English Scientific Research Paper Writing Assistant for Planning the outline of {field} English scientific research paper.
Generate a title and a brief outline for a {field} English scientific research paper on '{topic}'.
Please output in a dictionary format, structured as {{"title": "<Paper Title>", "outline": {{}}}}.
Within the "outline" dictionary, each key should be a chapter title, and each chapter title's corresponding value should be a python list of key points for that chapter.
"""
        out_prompt = PromptTemplate(
            template=outline_template,
            input_variables=['field', 'topic']
        )

        llm_outline_chain = LLMChain(
            llm=llm,
            prompt=out_prompt,
            verbose=True,
        )
        result = eval(llm_outline_chain.run({
            'field': field,
            'topic': topic,
        }))
        print(result)
        return result


def revise_text_test1(field, topic, section, outlines, draft, ref_path, related_works):
    from langchain.document_loaders import PyPDFLoader
    from langchain.chains.summarize import load_summarize_chain

    points = outlines['outline'][section]['points']
    outline = print_section_outline(outlines, section)
    # related_work = ''
    # for work in related_works:
    #     related_work += work[0]
    #     related_work += '\n'
    #     related_work += work[2][section]
    #     related_work += '\n\n'
    # 加载参考文献并总结主要内容
    references = ''
    loaders = []
    pdf_list = os.listdir(ref_path)
    pdf_list = [pdf for pdf in pdf_list if pdf != 'chroma']
    for pdf in pdf_list:
        loaders.append(PyPDFLoader(os.path.join(ref_path, pdf)))
    llm = OpenAI(
        model_name='gpt-3.5-turbo-instruct',
        temperature=0
    )
    llm_summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        output_key='summary',
        verbose=True
    )
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo-16k-0613',
        temperature=0
    )

    if len(loaders) > 0:
        for loader in loaders:
            pdfname = loader.file_path.split('/')
            pdfname = pdfname[len(pdfname) - 1]
            docs = loader.load_and_split()
            summary = llm_summarize_chain.run(docs)
            references = references + 'filename: ' + pdfname + '\n' + summary + '\n\n'

    text = ''
    for i in range(len(points)):
        para_template = """
You are a helpful English scientific research paper writing assistant.
You will be given the {section} section of the related paper, references list, the outline of {section} section, the draft about '{topic}' and the finished text of {section} in this paper.
Continue to write one paragraph to discuss '{point}'.
Select files in REFERENCE LIST and cite them at the relevant sentences in the generated paragraph.
Focus on the contextual coherence and avoid writing redundant content with other key points.
Keep paragraph length between 50-150 words.

# OUTLINE
{outline}

# REFERENCE LIST
{references}

# DRAFT
{draft}

#FINISHED TEXT
{text}

# ANSWER
"""

        para_prompt = PromptTemplate(
            template=para_template,
            input_variables=["field", "topic", "section", "outline", "point", "draft", "references",
                             "text"],
        )
        llm_para_chain = LLMChain(
            llm=llm,
            prompt=para_prompt,
            verbose=True,
        )
        result = llm_para_chain.run({
            "field": field,
            "topic": topic,
            "section": section,
            "outline": outline,
            "point": points[i],
            "draft": draft[i],
            "references": references,
            "text": text
        })
        text += result
        text += '\n\n'
#     para_template = """
# You are a helpful English scientific research paper writing assistant.
# You are trying to write the {section} section of the {field} English scientific research paper about '{topic}'.
# You will be given the {section} section of the related papers, the references, the outline of the {section} section and the draft of each point in the {section} section.
# You should imitate the language style of {section} section of the related paper.
# Cite the used filename at corresponding places in the generated paragraph.
# Focus on the contextual coherence and avoid writing redundant content with other key points.
# Keep each paragraph length between 50-150 words.
#
# # OUTLINE
# {outline}
#
# # RELATED PAPERS
# {related_works}
#
# # REFERENCES
# {references}
#
# # DRAFT
# {draft}
#
# # ANSWER
# """
#     draft_ = ''
#     for i in range(len(draft)):
#         draft_ += points[i] + '\n'
#         draft_ += draft[i] + '\n\n'
#     print(draft_)
#     draft = draft_
#     para_prompt = PromptTemplate(
#         template=para_template,
#         input_variables=["field", "topic", "section", "outline", "draft", "related_works", "references"]
#     )
#     llm_para_chain = LLMChain(
#         llm=llm,
#         prompt=para_prompt,
#         verbose=True,
#     )
#     text = llm_para_chain.run({
#         "field": field,
#         "topic": topic,
#         "section": section,
#         "outline": outline,
#         "draft": draft,
#         "related_works": related_work,
#         "references": references,
#     })
    print(text)
    return text


if __name__ == '__main__':
    path = '/Users/luqi/PycharmProjects/writing/users/luqi_0123/related_works'
    related_works = [('CASES: A Cognition-Aware Smart Eyewear System for Understanding How People '
  'Read',
  ['Introduction',
   'Related Work',
   'Method',
   'Experiment and Experiment and Result',
   'Discussion'],
  {'Discussion': '',
   'Experiment and Experiment and Result': ' 458\n'
                                           'This section describes experiments '
                                           'to evaluate CASES, the '
                                           'cognition-aware eyewear system for '
                                           'estimating 459\n'
                                           'reading states. We first detail '
                                           'the experimental setup, data '
                                           'collection, and evaluation '
                                           'measures. We present 460\n'
                                           'results and quantify the technical '
                                           'capabilities of CASES. All '
                                           'experimental procedures are '
                                           'approved by the ethical 461\n'
                                           'committee at our University. 462\n'
                                           '5.1 Evaluation Methodology 463\n'
                                           '5.1.1 Experimental Setup. We '
                                           'recruited 25 participants by '
                                           'posting a questionnaire at our '
                                           'university campus. 464\n'
                                           'Specifically, we distributed '
                                           'informed consent forms for the '
                                           'participants before the experiment '
                                           'started. In the 465\n'
                                           'consent form, we informed them '
                                           'about the purpose of our study and '
                                           'the procedure of the experiments '
                                           'and gave 466\n'
                                           'them the option to withdraw at any '
                                           'time during the experiments. All '
                                           'participants signed the informed '
                                           'consent 467\n'
                                           'form. After completing their '
                                           'sessions, the subjects received '
                                           'either local currency equivalent '
                                           'to 14 dollars or a 468\n'
                                           'thank-you gift worth approximately '
                                           '14 dollars for their '
                                           'participation. A summary of the '
                                           'participant demographics 469\n'
                                           'follows. 470\n'
                                           '•Age: 22–28 years old with an '
                                           'average age of 23.5, 471\n'
                                           '•Gender ratio : 19 males (76.0%) '
                                           'and 6 females (24.0%). 472\n'
                                           '•Native/non-native speaker : 5 '
                                           'native speakers (20.0%) and 20 '
                                           'non-native ones (80.0%). 473\n'
                                           'As shown in Figure 6 (a), the '
                                           'participant wears eyeglasses and '
                                           'sits in front of the computer to '
                                           'read. While reading, 474\n'
                                           'we record videos using the eye '
                                           'camera and time-aligned videos '
                                           'using the scene camera. 475\n'
                                           ' There is something almost '
                                           'delightful in the detachment from\n'
                                           'reality of advertisements showing '
                                           'mass-produced cars marketed\n'
                                           'as symbols of individuality and of '
                                           'freedom when most of their\n'
                                           'lives will be spent making short '
                                           'journeys on choked roads.\n'
                                           ' ... round it.5\n'
                                           'liberation\n'
                                           '(a) A participant is reading (b)  '
                                           'Screenshot from the labeling tool\n'
                                           'Fig. 6. The in-lab setting of '
                                           'CASES experimental study.\n'
                                           '5.1.2 Text Material Selection. '
                                           'Texts should cover a wide range of '
                                           'subjects so readers can enter '
                                           'multiple reading 476\n'
                                           'states. Moreover, each text should '
                                           'be short, allowing participants to '
                                           'read several texts. This study '
                                           'selects 36 477\n'
                                           'articles with the following three '
                                           'subjects: 478\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.CASES: A Cognition-Aware '
                                           'Smart Eyewear System for '
                                           'Understanding How People Read '
                                           '•0:15\n'
                                           'Subject matter 1 One-minute BBC '
                                           'world news4: 10 articles with '
                                           'approximately 300 words per '
                                           'article on 479\n'
                                           'average. 480\n'
                                           'Subject matter 2 English '
                                           'qualification tests5: 16 articles '
                                           'containing reading comprehension '
                                           'materials 481\n'
                                           'with approximately 450 words per '
                                           'article on average. 482\n'
                                           'Subject matter 3 Philosophy '
                                           'related [ 63]: 10 articles with '
                                           'approximately 500 words per '
                                           'article on average. 483\n'
                                           'The first two of these provide '
                                           'challenging words and sentences, '
                                           'respectively. The third may lead '
                                           'to mind 484\n'
                                           'wandering. We anticipate that most '
                                           'participants are unfamiliar with '
                                           'the third subject matter, and it '
                                           'is hard to 485\n'
                                           'understand the content without '
                                           'prior knowledge. The idea of '
                                           'mundane subject selection to '
                                           'introduce mind 486\n'
                                           'wandering follows a recent work '
                                           '[54]. 487\n'
                                           'Considering the diverse '
                                           'backgrounds and prior knowledge of '
                                           'various participants, texts should '
                                           'also cover a 488\n'
                                           'wide range of subject classes. '
                                           'According to Dewey Decimal '
                                           'Classification (DDC) method [ 70], '
                                           'we categorize 489\n'
                                           'the selected articles into ten '
                                           'subject classes, including “social '
                                           'science”, “religion”, and eight '
                                           'other subjects. Prior 490\n'
                                           'to data collection, we select an '
                                           'approximately equal number of '
                                           'articles from each topic class, '
                                           'except for the 491\n'
                                           'philosophy articles. 492\n'
                                           '5.1.3 Dataset. 493\n'
                                           '(1) Data Collection. The CASES '
                                           'requires time-aligned eye gaze '
                                           'data and text data (i.e., the '
                                           'words or sentences 494\n'
                                           'being read) to detect reading '
                                           'states. In addition, the '
                                           'synchronized data should capture '
                                           'continuous reading, during 495\n'
                                           'which users may encounter various '
                                           'reading states. To the best of our '
                                           'knowledge, there are no publically '
                                           'available 496\n'
                                           'datasets suitable for our problem. '
                                           'Therefore, we first develop an '
                                           'online system to collect data '
                                           'meeting our 497\n'
                                           'requirements. To facilitate '
                                           'research, we release the collected '
                                           'dataset, which is online '
                                           'available.6498\n'
                                           'Prior to data collection, we '
                                           'select an approximately equal '
                                           'number of articles from each topic '
                                           'class, except 499\n'
                                           'for the philosophy articles. Then, '
                                           'these articles are randomly '
                                           'assigned to each participant. '
                                           'Then, each article 500\n'
                                           'is divided into pages. There are '
                                           'around 240 words per page in '
                                           'single-spaced 18-point typeface. '
                                           'After that, we 501\n'
                                           'randomly select articles from each '
                                           'topic for the participants to '
                                           'ensure that they cover all three '
                                           'subject matters. 502\n'
                                           'This design allows most '
                                           'participants to encounter numerous '
                                           'reading states. Each article is '
                                           'read by an average of 503\n'
                                           'five participants. We verbally '
                                           'instruct participants on how to '
                                           'use the data-collection system, '
                                           'such as navigating 504\n'
                                           'to the next/previous page. '
                                           'Finally, each participant reads '
                                           'the texts. Reading one article '
                                           'takes approximately six 505\n'
                                           'minutes. 506\n'
                                           '(2) Ground-Truth Labeling. After '
                                           'completing an article, the '
                                           'participant is immediately '
                                           'instructed to label their 507\n'
                                           'reading states. We developed a '
                                           'labeling tool with a GUI window to '
                                           'accelerate labeling. Participants '
                                           'can review 508\n'
                                           'each page of the article. On each '
                                           'page, they use a single click to '
                                           'label the words they cannot '
                                           'comprehend and 509\n'
                                           'use a double-click to label '
                                           'sentences they do not comprehend. '
                                           'We also provide a button at the '
                                           'right top of each 510\n'
                                           'sentence for users to mark whether '
                                           'their minds wandered when reading '
                                           'it. The annotated words and '
                                           'sentences 511\n'
                                           'are highlighted in different '
                                           'colors so users can quickly '
                                           'double-check their annotations, as '
                                           'shown in Figure 6 (b). 512\n'
                                           'Annotating one article takes '
                                           'around three minutes. In total, '
                                           'the data collection process, '
                                           'including the annotation 513\n'
                                           'collection, took us approximately '
                                           'fourteen days. 514\n'
                                           '(3) Dataset Statistics. The '
                                           'collected dataset is randomly '
                                           'split into training (80%) and test '
                                           '(20%) sets per par- 515\n'
                                           'ticipant/article. The total '
                                           'numbers of labels for “word-level '
                                           'processing '
                                           'difficulties”/“sentence-level '
                                           'processing 516\n'
                                           'difficulties”/“mind wandering” are '
                                           '1005/244/200. 517\n'
                                           '4https://www.bbc.com/news\n'
                                           '5https://cet.neea.edu.cn/\n'
                                           '6https://drive.google.com/drive/folders/1AZmL1YhUU49ZOmCJKxqWsQUVFIW5nedo?usp=sharing\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.0:16 •Anonymous Authors\n'
                                           'Social\n'
                                           'scienceReligion Computer\n'
                                           'science,\n'
                                           'information and\n'
                                           'general worksLanguage Philosophy '
                                           'and\n'
                                           'psychologyScience Technology Arts '
                                           'and\n'
                                           'recreationLiterature History and\n'
                                           'geography\n'
                                           'Topic class012345ScoreInterest '
                                           'level (mean=2.78, std=1.18)\n'
                                           'Familiarity (mean=2.15, std=0.93)\n'
                                           'Fig. 7. Score of interest level '
                                           'and familiarity for each topic '
                                           'class.\n'
                                           'We survey the participants’ '
                                           'interest level and familiarity '
                                           'with the ten topics to further '
                                           'verify the fairness of the 518\n'
                                           'selected topics, i.e. we expect '
                                           'that the interest level and '
                                           'familiarity are evenly distributed '
                                           'across all topics. Using 519\n'
                                           'the Likert scale7, we asked '
                                           'participants to score their '
                                           'interest level and familiarity '
                                           'with the articles they read. 520\n'
                                           'The scale ranges from 1 to 5; a '
                                           'higher score indicates that a '
                                           'participant is more familiar with '
                                           'or more interested 521\n'
                                           'in the article. As shown in Figure '
                                           '7, participants gave roughly '
                                           'similar interest scores (mean = '
                                           '2.78, std = 1.18) and 522\n'
                                           'familiarity (mean = 2.15, std = '
                                           '0.93) on the ten topic classes, '
                                           'indicating that the ten topic '
                                           'classes have covered 523\n'
                                           'individual participants evenly. '
                                           'The means of interest level and '
                                           'familiarity of all topics are '
                                           'around 2.5, suggesting 524\n'
                                           'that the topics are intermediate '
                                           'to participants. 525\n'
                                           'We also visualize the distribution '
                                           'of the average number of labels '
                                           'per article at various levels of '
                                           'interest and 526\n'
                                           'familiarity in Figure 8 (left). It '
                                           'is clear that readers give '
                                           'approximately the same number of '
                                           'labels per article 527\n'
                                           'under each interest level. Also, '
                                           'Figure 8 (right) shows that the '
                                           'number of labels per article '
                                           'decreases as familiarity 528\n'
                                           'increases. This is in line with '
                                           'our intuition, as participants '
                                           'often give more labels for their '
                                           'unfamiliar articles. 529\n'
                                           '5.1.4 Evaluation Metrics. Because '
                                           'our framework is hierarchical and '
                                           'multi-task, we need to adopt '
                                           'appropriate 530\n'
                                           'measures to evaluate each task. '
                                           'The first task is binary '
                                           'classification of whether a reader '
                                           'is facing difficulty 531\n'
                                           'processing a word. We evaluate its '
                                           'performance using accuracy and the '
                                           'receiver operating characteristics '
                                           '(ROC) 532\n'
                                           'curve. The second task is '
                                           'hierarchical multi-label '
                                           'classification at the sentence '
                                           'level, which includes '
                                           'sentence-level 533\n'
                                           'Task I and Task II. Task II is '
                                           'multi-labeled, following previous '
                                           'work [ 88]. We therefore use the '
                                           'multilabel-based 534\n'
                                           'macro-averaging metric, i.e., '
                                           'averaged-accuracy and ROC curve, '
                                           'to evaluate it. 535\n'
                                           '5.1.5 Baseline methods. We conduct '
                                           'ablation studies to evaluate '
                                           'CASES, as there is no prior work '
                                           'solving the 536\n'
                                           'problem addressed in this work, '
                                           'thus making direct comparisons '
                                           'with prior work infeasible. We use '
                                           'the following 537\n'
                                           'three baseline methods for '
                                           'evaluation. 538\n'
                                           '(1)Visual: Previous studies have '
                                           'demonstrated that some reading '
                                           'states, such as mind wandering, '
                                           'can be 539\n'
                                           'identified using gaze-relevant '
                                           'features [ 19,54], which are '
                                           'closely related to our work. To '
                                           'validate whether 540\n'
                                           'the eye-relevant features are '
                                           'sufficient for reading state '
                                           'recognition at multiple text '
                                           'element granularities 541\n'
                                           '(words and sentences), this work '
                                           'uses baseline method leveraging 13 '
                                           'eye-relevant features (9 '
                                           'word-level 542\n'
                                           'features and 4 sentence-level '
                                           'features described in Section '
                                           '4.1.3) to identify the state while '
                                           'reading. We 543\n'
                                           'use the support vector machine '
                                           '(SVM) method to conduct the three '
                                           'classification tasks: word-level '
                                           'task, 544\n'
                                           'sentence-level Task I, and '
                                           'sentence-level Task II. This work '
                                           'adopts SVM as it has been '
                                           'successfully applied 545\n'
                                           '7https://en.wikipedia.org/wiki/Likert_scale\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.CASES: A Cognition-Aware '
                                           'Smart Eyewear System for '
                                           'Understanding How People Read '
                                           '•0:17\n'
                                           '1 2 3 4 5\n'
                                           'Interest level024681012# of '
                                           'labels/article on average\n'
                                           'Word-level processing difficulty\n'
                                           'Sentence-level processing '
                                           'difficulty\n'
                                           'Mind wandering\n'
                                           '1 2 3 4 5\n'
                                           'Familiarity024681012# of '
                                           'labels/article on average\n'
                                           'Word-level processing difficulty\n'
                                           'Sentence-level processing '
                                           'difficulty\n'
                                           'Mind wandering\n'
                                           'Fig. 8. Number of labels per '
                                           'article on average under each '
                                           'interest level and familiarity.\n'
                                           'to various classification tasks [ '
                                           '78], and is one of the widely used '
                                           'methods in similar tasks [ 20,54]. '
                                           'For 546\n'
                                           'simplicity, we refer to this '
                                           'method as Visual . 547\n'
                                           '(2)Visual+: Eye movement patterns '
                                           'are good indicators for reading '
                                           'state recognition. Inspired by '
                                           'prior 548\n'
                                           'work [ 75] that leverages deep '
                                           'neural network (DNN) to achieve '
                                           'accurate eye movement pattern '
                                           'identi- 549\n'
                                           'fication, we use the 8-dimensional '
                                           'deep features extracted from a '
                                           'deep neural network (1D-CNN with '
                                           '550\n'
                                           'BLSTM [ 75]) to improve the '
                                           'accuracy of reading state '
                                           'estimation. To make a fair '
                                           'comparison, the extracted 551\n'
                                           'deep features are concatenated '
                                           'with the 13 expert-designed '
                                           'features and sent to the CAE '
                                           'module to estimate 552\n'
                                           'reading state. This baseline '
                                           'method is an improved version of '
                                           'the Visual method called Visual+ . '
                                           '553\n'
                                           '(3)NLP: Visual and Visual+ '
                                           'identify reading states based '
                                           'solely on visual attention '
                                           'features. To verify the 554\n'
                                           'classification performance based '
                                           'on the semantic content of texts, '
                                           'we designed this baseline method, '
                                           'dubbed 555\n'
                                           'NLP. As in the Visual+ method, we '
                                           'first extract semantic features '
                                           'using the SAE module and then send '
                                           'the 556\n'
                                           'extracted features to the CAE '
                                           'module to infer reading states. '
                                           '557\n'
                                           '5.2 Results 558\n'
                                           '5.2.1 Overall Performance. Figure '
                                           '9 shows the reading state '
                                           'recognition performance of our '
                                           'methods and three 559\n'
                                           'baseline methods. The proposed '
                                           'method achieves the best '
                                           'performance. Compared with the '
                                           'Visual method, i.e., 560\n'
                                           'conventional eye-tracking only, '
                                           'CASES improve the accuracy by '
                                           '6.85%, 8.55%, 20.90% for the '
                                           'word-level task and 561\n'
                                           'the sentence-level Task I and Task '
                                           'II. Furthermore, compared with the '
                                           'baseline method Visual+ and NLP, '
                                           'CASES 562\n'
                                           'has superior reading state '
                                           'estimation. For example, the '
                                           'sentence-level Task II detection '
                                           'accuracy of CASES is 563\n'
                                           '86.64% while it is 79.15% or lower '
                                           'for the baseline methods. We '
                                           'conclude that using context '
                                           'derived from text 564\n'
                                           'improves reading state estimation. '
                                           '565\n'
                                           'We plot the Receiver Operating '
                                           'Characteristic (ROC) of different '
                                           'methods. Figures 10a, 10b, and 10c '
                                           'demonstrate 566\n'
                                           'that CASES outperforms the '
                                           'baseline methods in Area Under the '
                                           'Curve (AUC), which is one of the '
                                           'most widely 567\n'
                                           'used performance measures in '
                                           'classification or retrieval '
                                           'problems. 568\n'
                                           'The next section further explains '
                                           'why CASES outperforms the baseline '
                                           'methods and how it offers semantic '
                                           '569\n'
                                           'explanations of the predicted '
                                           'reading states. 570\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.0:18 •Anonymous Authors\n'
                                           'Word-level task Sentence-level '
                                           'task I Sentence-level task '
                                           'II0.00.20.40.60.81.0Accuracy0.93\n'
                                           '0.830.870.92\n'
                                           '0.75 0.750.87\n'
                                           '0.76\n'
                                           '0.720.89\n'
                                           '0.730.79\n'
                                           'Ours\n'
                                           'NLP\n'
                                           'Visual\n'
                                           'Visual+\n'
                                           'Fig. 9. Reading state '
                                           'classification accuracy for CASES '
                                           'and the baseline methods.\n'
                                           '0.0 0.2 0.4 0.6 0.8 1.0\n'
                                           'False Positive '
                                           'Rate0.00.20.40.60.81.0True '
                                           'Positive RateOurs (area = 0.96)\n'
                                           'NLP (area = 0.94)\n'
                                           'Visual+ (area = 0.82)\n'
                                           'Visual (area = 0.77)\n'
                                           '(a) Word-level task\n'
                                           '0.0 0.2 0.4 0.6 0.8 1.0\n'
                                           'False Positive '
                                           'Rate0.00.20.40.60.81.0True '
                                           'Positive RateOurs (area = 0.85)\n'
                                           'NLP (area = 0.77)\n'
                                           'Visual+ (area = 0.82)\n'
                                           'Visual (area = 0.74) (b) '
                                           'Sentence-level task I\n'
                                           '0.0 0.2 0.4 0.6 0.8 1.0\n'
                                           'False Positive '
                                           'Rate0.00.20.40.60.81.0True '
                                           'Positive RateOurs (AUC = 0.82)\n'
                                           'NLP (AUC = 0.71)\n'
                                           'Visual+ (AUC = 0.80)\n'
                                           'Visual (AUC = 0.71) (c) '
                                           'Sentence-level task II\n'
                                           'Fig. 10. ROC for CASES and the '
                                           'baseline methods.\n'
                                           '\n'
                                           ' 571\n'
                                           'This work aims to study '
                                           'progression through cognitive '
                                           'states while reading to assist our '
                                           'understanding of the 572\n'
                                           'reading process. To this end, we '
                                           'have conducted in-field pilot '
                                           'studies using CASES, the proposed '
                                           'system, for 573\n'
                                           'totally three and a half months. '
                                           'This section first summarizes the '
                                           'initial findings around our the '
                                           'designed two 574\n'
                                           'RQ and hypotheses using CASES. '
                                           'Then, it demonstrates the '
                                           'capability of EYEReader to make '
                                           'helpful real-time 575\n'
                                           'interventions when reading '
                                           'difficulties are encountered. '
                                           'Finally, it describes the '
                                           'limitations of our system and 576\n'
                                           'indicates possible extensions of '
                                           'this work. 577\n'
                                           '6.1 The Procedure of the Pilot '
                                           'Study 578\n'
                                           'We recruited thirteen volunteers '
                                           'to participate in our pilot study '
                                           'from our University. The average '
                                           'age is 23.9 579\n'
                                           'years (SD=1.6, min=22, max=28), '
                                           'with n=4 (30.8%) female and n=9 '
                                           '(69.2%) males. There are 10 '
                                           'non-native readers 580\n'
                                           '(76.9%) and 3 native ones (23.1%). '
                                           'The non-native participants '
                                           'reported that they have passed the '
                                           'college English 581\n'
                                           'test and the native readers are '
                                           'college-level students at our '
                                           'university. 582\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.CASES: A Cognition-Aware '
                                           'Smart Eyewear System for '
                                           'Understanding How People Read '
                                           '•0:19\n'
                                           'During the pilot study, '
                                           'participants wore the prototype '
                                           'eyeglasses, sat in front of the '
                                           'computer, and logged in to 583\n'
                                           'the development website to read. '
                                           'The participants can either take '
                                           'the eyeglasses with them and use '
                                           'the eyeglasses 584\n'
                                           'whenever they would like to do the '
                                           'experiments, or come to our '
                                           'laboratory for the experiments. '
                                           'Participants are 585\n'
                                           'encouraged to use the system '
                                           'whenever they read, as reasonable '
                                           'observations require the prolonged '
                                           'engagement 586\n'
                                           'of participants. 587\n'
                                           'The pilot study lasts three and a '
                                           'half months and consisted of two '
                                           'stages. During the first stage, we '
                                           'require 588\n'
                                           'participants to label the words '
                                           'and sentences they encountered '
                                           'difficulty processing, and these '
                                           'labels are treated 589\n'
                                           'as ground truth. Based on the '
                                           'qualitative evaluation [ 68], we '
                                           'examine the labelled data point by '
                                           'point at different 590\n'
                                           'granularities around the designed '
                                           'RQ and hypotheses. Then, we make '
                                           'several findings on how people '
                                           'read at 591\n'
                                           'different granularities, i.e., '
                                           'single words and sentences, and '
                                           'summarized the following six '
                                           'patterns to discuss. The 592\n'
                                           'second stage focuses on applying '
                                           'EYEReader in practice. At the end '
                                           'of the pilot study, each '
                                           'participant completes 593\n'
                                           'a survey of their opinions on the '
                                           'usability and value of EYEReader. '
                                           'Finally, we confirme the proposed '
                                           'hypotheses. 594\n'
                                           '6.2 Key Observations 595\n'
                                           '6.2.1 Observations at the Word '
                                           'Level. In this section, we present '
                                           'three observations on how users '
                                           'read at the 596\n'
                                           'single-word level. 597\n'
                                           'Observation I: Users comprehend '
                                           'the lexical meanings of words by '
                                           'directing their gazes more '
                                           'frequently toward 598\n'
                                           'material they find difficult to '
                                           'process. When users encounter '
                                           'difficulty processing a word, they '
                                           'usually gaze at 599\n'
                                           'it longer, and more times than '
                                           'typical. This observation is '
                                           'consistent with prior evidence '
                                           'about the process of 600\n'
                                           'comprehending single words during '
                                           'reading [ 16,53]. Figure 11 '
                                           'illustrates one example of this '
                                           'observation, where 601\n'
                                           'participant P6 has difficulty '
                                           'comprehending the meaning of '
                                           '“mitigate” and “debris”. P6 '
                                           'fixates “mitigate” (fixation 602\n'
                                           'label 13) and “debris” (with '
                                           'fixation label 19) for a long time '
                                           'and reads them more than two '
                                           'times. In particular, 603\n'
                                           'P6 has the longest fixation '
                                           'duration on the word “debris” and '
                                           'he has the most reading times on '
                                           'the word “debris” 604\n'
                                           'and “mitigate”.\n'
                                           'Fig. 11. Visualization of visual '
                                           'attention for P6 reading a '
                                           'sentence. Each circle represents a '
                                           'fixation point. The larger the '
                                           'area\n'
                                           'of the circle, the longer the '
                                           'fixation duration. The circle '
                                           'number denotes the timestamp of '
                                           'the fixation time. Top: Raw text;\n'
                                           'Bottom: Text with filtered point '
                                           'of gazes\n'
                                           '605\n'
                                           'Figure 12 provides another example '
                                           'of this observation for a native '
                                           'participant. Participant '
                                           'P12∗(∗indicates 606\n'
                                           'native user here in after) is '
                                           'facing challenging words '
                                           '“counterbalanced” (with fixation '
                                           'label 10) and “sketch” 607\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.0:20 •Anonymous Authors\n'
                                           '(fixation labels 19 and 21). Under '
                                           'the text context presented in '
                                           'Figure 12, we observe that P12∗has '
                                           'the most 608\n'
                                           'prolonged fixation duration on '
                                           'words “sketch” and '
                                           '“counterbalanced”. 609\n'
                                           'Fig. 12. Visualization of visual '
                                           'attention for P12∗reading a '
                                           'sentence. Top: Raw text; Bottom: '
                                           'Text with filtered point of '
                                           'gazes.\n'
                                           'Observation II: When a user '
                                           'encounters difficulty processing a '
                                           'word, the user first directs their '
                                           'gaze to the word and 610\n'
                                           'then to other words to examine the '
                                           'semantic context. Readers '
                                           'generally avoid breaking their '
                                           'chain of thinking by 611\n'
                                           'stopping when a difficult word is '
                                           'encountered, especially when the '
                                           'word does not affect their '
                                           'understanding of 612\n'
                                           'the text. However, when readers '
                                           'consider a difficult word to be '
                                           'highly topic-relevant or '
                                           'meaningful for subsequent 613\n'
                                           'text comprehension, they tend to '
                                           'interrupt their reading and '
                                           'attempt to deduce the semantic '
                                           'meaning of the 614\n'
                                           'word from its semantic context. '
                                           'This observation differs from a '
                                           'previous study [ 16] and our next '
                                           'observation 615\n'
                                           'complements it. 616\n'
                                           'Observation III: When users '
                                           'examine the semantic context of a '
                                           'difficult-to-process word, they '
                                           'gather semantic 617\n'
                                           'clues by shifting their gazes to '
                                           'different locations even when '
                                           'considering the same difficult '
                                           'word, from the same text, 618\n'
                                           'under similar reading conditions. '
                                           'Readers typically attempt to find '
                                           'an appropriate location in the '
                                           'text to help 619\n'
                                           'comprehend the current '
                                           'difficult-to-process word. The '
                                           'text at the location should reveal '
                                           'the relevant information 620\n'
                                           'about the difficult word. Also, '
                                           'that location varies from person '
                                           'to person, depending on their '
                                           'current cognitive 621\n'
                                           'states about the context. 622\n'
                                           'Figure 13 shows the proportion of '
                                           'the three above observations for '
                                           'each participant by summarizing '
                                           'their past 623\n'
                                           'experienced processing difficult '
                                           'words. We observe that the ten '
                                           'non-native participants experience '
                                           'Observation I 624\n'
                                           'in most cases (around 87.47% cases '
                                           'on average), and they fall into '
                                           'Observation II & Observation III '
                                           'in fewer times, 625\n'
                                           'i.e., around 12.53% on average. In '
                                           'contrast, the native participants '
                                           'experience Observation II & '
                                           'Observation III in 626\n'
                                           'most cases (around 60.67% cases on '
                                           'average), and they fall into '
                                           'Observation I fewer times, i.e., '
                                           'around 39.33% on 627\n'
                                           'average. This aligns with our '
                                           'intuition, as we anticipate that '
                                           'native readers are more adept at '
                                           'leveraging the 628\n'
                                           'context cues from texts to help '
                                           'their reading comprehension. 629\n'
                                           'Figure 14 shows an exemplary case '
                                           'to provide further insights on '
                                           'Observation II and Observation '
                                           'III. Here two 630\n'
                                           'readers, P2 and P5, face the same '
                                           'reading difficulty in '
                                           'comprehending the word '
                                           '“liberation” when they read the '
                                           '631\n'
                                           'same sentence from the same '
                                           'article. We can see that the two '
                                           'participants first direct their '
                                           'visual attention to the 632\n'
                                           'target word, “liberation” where '
                                           'the fixation labels are 14 and 13 '
                                           'for P4 and P5, respectively. They '
                                           'then shift their 633\n'
                                           'gazes. Participant P2 gazes back '
                                           'at the previously read word, '
                                           '“pleasant”, while Participant P5 '
                                           'gazes forward to 634\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.CASES: A Cognition-Aware '
                                           'Smart Eyewear System for '
                                           'Understanding How People Read '
                                           '•0:21\n'
                                           '/uni00000033/uni00000014 '
                                           '/uni00000033/uni00000015 '
                                           '/uni00000033/uni00000016 '
                                           '/uni00000033/uni00000017 '
                                           '/uni00000033/uni00000018 '
                                           '/uni00000033/uni00000019 '
                                           '/uni00000033/uni0000001a '
                                           '/uni00000033/uni0000001b '
                                           '/uni00000033/uni0000001c '
                                           '/uni00000033/uni00000014/uni00000013/uni00000033/uni00000014/uni00000014*/uni00000033/uni00000014/uni00000015*/uni00000033/uni00000014/uni00000016*/uni00000013/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001b/uni00000014/uni00000011/uni00000013/uni00000033/uni00000055/uni00000052/uni00000053/uni00000052/uni00000055/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000032/uni00000045/uni00000056/uni00000048/uni00000055/uni00000059/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002c\n'
                                           '/uni00000032/uni00000045/uni00000056/uni00000048/uni00000055/uni00000059/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002c/uni0000002c/uni00000003/uni00000009/uni00000003/uni0000002c/uni0000002c/uni0000002c\n'
                                           'Fig. 13. Profile of word-level '
                                           'comprehension failures for '
                                           'thirteen readers. ∗indicates '
                                           'native participants.\n'
                                           'the word, “promised”. Both of '
                                           'these words are semantically '
                                           'relevant to the difficult word, '
                                           '“liberation”, as shown 635\n'
                                           'in Figure 14 (top row). 636\n'
                                           'P2\n'
                                           'P5\n'
                                           'Semantic correlation\n'
                                           'Topic-relevant \n'
                                           'Difficult\n'
                                           'Fig. 14. An example in which two '
                                           'non-native users both have '
                                           'difficulty understanding the word, '
                                           '“liberation”.\n'
                                           'Figure 15 provides an example for '
                                           'two native readers, P12∗and P13∗. '
                                           'They face the same reading '
                                           'difficulty 637\n'
                                           'in comprehending the challenging '
                                           'word “realism”. Clearly, P12∗and '
                                           'P13∗first direct their gaze to '
                                           '“realism” 638\n'
                                           'with fixation labels 14 and 11, '
                                           'respectively, and then shift their '
                                           'gazes. P12∗gazes forward to an '
                                           'antonym word 639\n'
                                           '“antirealism”. Differently, '
                                           'P13∗gazes back at the already-read '
                                           'word “internal”, a modifier of the '
                                           'target word. 640\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.0:22 •Anonymous Authors\n'
                                           'P12*\n'
                                           'P13*Sentence\n'
                                           'Fig. 15. An example in which two '
                                           'native users both have difficulty '
                                           'understanding the word, '
                                           '“realism”.\n'
                                           '6.2.2 Observations at Sentence '
                                           'Level. This section focuses on two '
                                           'modes of comprehending sentences: '
                                           'interpretive 641\n'
                                           '(semantic) and structural '
                                           '(syntactic). 642\n'
                                           'Observation IV: People '
                                           'incrementally comprehend the '
                                           'semantics of a sentence as they '
                                           'read each word, while with 643\n'
                                           'different gaze time series. Figure '
                                           '16 shows the inter-reader '
                                           'differences in gaze time series '
                                           'for the same sentence. 644\n'
                                           'P1 focuses on the first parts of '
                                           'sentences (with more fixation, '
                                           'labels 0–12) while P4 focuses on '
                                           'other parts of 645\n'
                                           'sentences (fixation labels 9–11). '
                                           '646\n'
                                           'P4P1\n'
                                           'Fig. 16. An example of two '
                                           'non-native users in the same '
                                           'sentence but with different visual '
                                           'attention.\n'
                                           'Figure 17 shows the inter-readers '
                                           'differences for two native readers '
                                           '(P11∗and P13∗), in the gaze '
                                           'timeseries. 647\n'
                                           'They both read the sentence '
                                           'sequentially but with different '
                                           'visual focuses. P13∗focuses on the '
                                           'first parts of the 648\n'
                                           'sentence, e.g., with more distinct '
                                           'locations of focus, while '
                                           'P11∗focuses on other parts of '
                                           'sentences. 649\n'
                                           'Observation V: Readers enter the '
                                           '“rereading” or “reanalysis” state '
                                           'at different times when having '
                                           'difficulty with 650\n'
                                           'the same sentence, as illustrated '
                                           'in Figure 18. P8 backtracks 3–4 '
                                           'words (with a fixation label '
                                           'starting from 28) 651\n'
                                           'when reading the middle of the '
                                           'sentence, and then continues '
                                           'reading the sentence; while P3 '
                                           'rereads the sentence 652\n'
                                           'from the beginning when reading '
                                           'the middle of the sentence '
                                           '(fixation label 12). 653\n'
                                           'Figure 19 depicts such an example '
                                           'for native users. The two users '
                                           'both face challenges in '
                                           'comprehending 654\n'
                                           'the sentence “ As countless boards '
                                           'and ...... and overall '
                                           'performance. ” We observe that '
                                           'P11∗rereads the sentence 655\n'
                                           '(fixation label 23) right after '
                                           'completing the first-pass reading '
                                           '(fixation label 22); while '
                                           'P13∗rereads the sentence 656\n'
                                           'from the beginning of the sentence '
                                           '(fixation label 6) when finishing '
                                           'the middle of the sentence '
                                           '(fixation label 5). 657\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.CASES: A Cognition-Aware '
                                           'Smart Eyewear System for '
                                           'Understanding How People Read '
                                           '•0:23\n'
                                           'P1 1*\n'
                                           'P13*\n'
                                           'Fig. 17. An example of two native '
                                           'users in the same sentence but '
                                           'with different visual attention.\n'
                                           'P3P8\n'
                                           'Fig. 18. An example of two '
                                           'non-native users having different '
                                           '“reread” behaviors when in the '
                                           'same reading state: encountering\n'
                                           'comprehension difficulties on the '
                                           'sentence.\n'
                                           'P1 1*\n'
                                           'P13*\n'
                                           'Fig. 19. An example of two native '
                                           'users having different “rereading” '
                                           'behaviors when in the same reading '
                                           'state: encountering\n'
                                           'comprehension difficulties on the '
                                           'sentence.\n'
                                           'Observation VI: Different people '
                                           '“reread” the same sentence with '
                                           'different reading states. Figure '
                                           '20 shows two 658\n'
                                           'non-native users reading the same '
                                           'sentence twice. P1 gets distracted '
                                           '(i.e., enters the mind wandering '
                                           'state) during 659\n'
                                           'the first reading of the sentence '
                                           '(typical fixation labels 2, 6, and '
                                           '14); therefore, P1 spends more '
                                           'time and has more 660\n'
                                           'fixations on the sentence in the '
                                           'second reading (fixation labels '
                                           '21, 24, 26, 28) than in the first '
                                           'pass. In contrast, P4 661\n'
                                           'spends more time when reading the '
                                           'sentence the first time (fixation '
                                           'labels 3, 17, and 18), but he '
                                           'quickly skims it 662\n'
                                           'the second time (fixation labels '
                                           '26 and 37). 663\n'
                                           'Figure 21 shows two native users, '
                                           'P11∗and P12∗, reading the same '
                                           'sentence twice. They label their '
                                           'reading 664\n'
                                           'states as sentence-level '
                                           'processing difficulty and mind '
                                           'wandering, respectively. '
                                           'P11∗spends more time and more 665\n'
                                           'fixations when reading the '
                                           'sentence in the first pass '
                                           '(typical fixation labels 1, 5, and '
                                           '18) than in the second 666\n'
                                           'pass (typical fixation labels 26, '
                                           '29, and 36). Also, P11∗rereads the '
                                           'sentence after completing the next '
                                           'sentence 667\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.0:24 •Anonymous Authors\n'
                                           'P1\n'
                                           'P4\n'
                                           'Fig. 20. An example of two '
                                           'non-native users “rereading” the '
                                           'same sentence with different '
                                           'reading states. P1 gets '
                                           'distracted\n'
                                           'during the first reading and he '
                                           'then reread the sentence.\n'
                                           '(fixation label 25). Differently, '
                                           'P12∗gets distracted during the '
                                           'first pass of the sentence '
                                           '(typical fixation labels 668\n'
                                           '5-15); therefore, P12∗rereads the '
                                           'sentence with more time and has '
                                           'more fixations on the sentence in '
                                           'the second 669\n'
                                           'pass (fixation labels 18, 21, 26 '
                                           'and 27). 670\n'
                                           'P1 1*\n'
                                           'P12*\n'
                                           'Fig. 21. An example of two native '
                                           'users “reread” the same sentence '
                                           'with different reading states. '
                                           'P11∗encounters processing\n'
                                           'difficulty with the sentence, and '
                                           'then P11∗rereads the sentence; '
                                           'P12∗gets distracted during the '
                                           'first reading, and then P12∗\n'
                                           'rereads the sentence.\n'
                                           '6.3 Evaluation of EYEReader in '
                                           'Practice 671\n'
                                           'Section 5 shows that CASES-Net '
                                           'accurately detects reading states. '
                                           'This section evaluates the ability '
                                           'of EYEReader 672\n'
                                           'to promote reading comprehension '
                                           'by detecting reading states '
                                           'implying processing difficulties '
                                           'and making 673\n'
                                           'real-time interventions. 674\n'
                                           'To make quantitative assessment of '
                                           'EYEReader, we define reading '
                                           'comprehension improvement as '
                                           '(𝑠𝑝𝑎𝑠𝑡− 675\n'
                                           '𝑠𝑝𝑟𝑒𝑠𝑒𝑛𝑡)/𝑠𝑝𝑎𝑠𝑡, where 𝑠𝑝𝑟𝑒𝑠𝑒𝑛𝑡 '
                                           'and𝑠𝑝𝑎𝑠𝑡denote the number of '
                                           'challenging words or sentences at '
                                           'present and in 676\n'
                                           'the past, respectively. The higher '
                                           'the (𝑠𝑝𝑎𝑠𝑡−𝑠𝑝𝑟𝑒𝑠𝑒𝑛𝑡)/𝑠𝑝𝑎𝑠𝑡, the '
                                           'higher the reading comprehension '
                                           'improvement. 677\n'
                                           'This definition is used to '
                                           'identify challenging words and '
                                           'sentences. After pilot studies, we '
                                           'ask participants to 678\n'
                                           'indicate whether they still face '
                                           'challenges in comprehending these '
                                           'words and sentences. Figure 22 '
                                           'shows the 679\n'
                                           'results. All thirteen participants '
                                           'have positive reading gains, which '
                                           'means EYEReader is effective in '
                                           'helping users 680\n'
                                           'to overcome unfamiliar words and '
                                           'complex sentences. 681\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.CASES: A Cognition-Aware '
                                           'Smart Eyewear System for '
                                           'Understanding How People Read '
                                           '•0:25\n'
                                           'P1 P2 P3 P4 P5 P6 P7 P8 P9 '
                                           'P10P11*P12*P13*00.20.40.60.81.0 '
                                           'Reading GainWords\n'
                                           'Sentences\n'
                                           'Fig. 22. Profile of reading gain '
                                           'for 13 participants in pilot '
                                           'studies. ∗indicates native '
                                           'readers.\n'
                                           '6.4 Feedback from Participants '
                                           '682\n'
                                           'We designed several open-ended '
                                           'questionnaires to qualitatively '
                                           'evaluate EYEReader. Thirteen '
                                           'questionnaires 683\n'
                                           'were sent to participants, twelve '
                                           'of which were returned. Among '
                                           'them, 10/12 of the participants '
                                           'positively 684\n'
                                           'commented on word-level '
                                           'intervention. They believe that '
                                           'fine-grained intervention at the '
                                           'word level can precisely 685\n'
                                           'pinpoint the reading difficulties '
                                           'they are experiencing. There are '
                                           '9/12 of the participants found '
                                           'sentence-level 686\n'
                                           'intervention helpful. In '
                                           'particular, when facing '
                                           'challenging sentences with complex '
                                           'syntactic structures, it was 687\n'
                                           'difficult to comprehend the '
                                           'sentence even though they were '
                                           'familiar with all the words. In '
                                           'this case, EYEReader 688\n'
                                           'helped them overcome this reading '
                                           'difficulty by highlighting the '
                                           'sentence and explaining it. In '
                                           'addition, 9/12 of 689\n'
                                           'the participants found EYEReader '
                                           'valuable in reminding them when '
                                           'their minds wandered; these '
                                           'participants 690\n'
                                           'stated that they usually do not '
                                           'realize when they are distracted. '
                                           'Timely reminders can make their '
                                           'reading more 691\n'
                                           'focused and efficient. 692\n'
                                           'Furthermore, we collect '
                                           'participants’ opinions regarding '
                                           'whether the eyewear hardware will '
                                           'negatively affect 693\n'
                                           'their reading process. Through '
                                           'conducting a questionnaire, we ask '
                                           'the participants in the pilot '
                                           'study to give 694\n'
                                           'scores for the comfortable level '
                                           'of hardware on a scale of 1-5; the '
                                           'corresponding description is '
                                           'listed in Table 1. A 695\n'
                                           'total of 13 questionnaires are '
                                           'sent out, and 12 are recalled. '
                                           'Statistical results show that '
                                           'participants generally 696\n'
                                           'think the hardware has a media or '
                                           'negligible impact on their reading '
                                           'process (mean = 3.33, std = 0.75). '
                                           '697\n'
                                           'In addition, once the trained '
                                           'CASES is applied to practice, we '
                                           'design the proper system '
                                           'intervention so as not 698\n'
                                           'to break the chain of thoughts of '
                                           'users. That is, the intervention '
                                           'can help readers avoid '
                                           'interrupting reading due 699\n'
                                           'to the encountered processing '
                                           'difficulties that lead them to '
                                           'seek help from other means [ 32], '
                                           'such as a dictionary. 700\n'
                                           'Therefore, we design the '
                                           'intervention process with minimal '
                                           'interaction cost and encourage '
                                           'readers to focus 701\n'
                                           'on the current reading. To assess '
                                           'the impact of the intervention on '
                                           'reading, we also conduct a '
                                           'questionnaire 702\n'
                                           'to collect readers’ opinions '
                                           'regarding the user-friendliness of '
                                           'intervention interaction. '
                                           'Similarly, we ask the 703\n'
                                           'participants in the pilot studies '
                                           'to give scores on a scale of 1-5; '
                                           'the corresponding description is '
                                           'listed in Table 2. 704\n'
                                           'Statistical results show that most '
                                           'participants generally think the '
                                           'intervention process is '
                                           'user-friendly (mean = 705\n'
                                           '3.92, std = 0.76). 706\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.0:26 •Anonymous Authors\n'
                                           'Table 1. Rating scale regarding '
                                           'the comfort of the\n'
                                           'hardware that does not have a '
                                           'negative impact on\n'
                                           'the reading process.\n'
                                           'Score Description\n'
                                           '1 Severe impact\n'
                                           '2 Significant impact\n'
                                           '3 Neutral\n'
                                           '4 Negligible impact\n'
                                           '5 No impact totallyTable 2. Rating '
                                           'scale used to describe whether the '
                                           'inter-\n'
                                           'vention is user-friendly that does '
                                           'not affect the reading\n'
                                           'process negatively.\n'
                                           'Score Description\n'
                                           '1 Very unfriendly\n'
                                           '2 Unfriendly\n'
                                           '3 Neutral\n'
                                           '4 Friendly\n'
                                           '5 Very friendly707\n'
                                           '6.5 Discussion and Future Work '
                                           '708\n'
                                           'CASES has the goal of accurately '
                                           'estimating and providing semantic '
                                           'explanations of reading states '
                                           'over time, 709\n'
                                           'which can facilitate the '
                                           'scientific study of reading by '
                                           'enabling a deeper understanding of '
                                           'the cognitive processes 710\n'
                                           'involved in learning to read, '
                                           'disentangling the complex '
                                           'combination of cognitive skills '
                                           'and their impact on 711\n'
                                           'reading fluency, and measuring the '
                                           'efficacy of methods for teaching '
                                           'reading and beneficial reading '
                                           'habits. 712\n'
                                           'This section first revisits the '
                                           'proposed research questions and '
                                           'hypotheses. Then, it briefly '
                                           'discusses the potential 713\n'
                                           'future works that will improve '
                                           'CASES. 714\n'
                                           '6.5.1 Revisiting Research '
                                           'Questions and Hypotheses. We '
                                           'confirm the hypotheses for the two '
                                           'presented research 715\n'
                                           'questions (RQ) based on the '
                                           'results and observations, which we '
                                           'detail below. 716\n'
                                           'RQ1: Do readers in the same '
                                           'reading states show different '
                                           'visual attention distributions on '
                                           'the reading text? 717\n'
                                           'Confirming hypothesis 1: Readers '
                                           'in the same reading state do show '
                                           'varying visual attention '
                                           'histories. As 718\n'
                                           'inter-person variation, i.e., '
                                           'individual difference, is '
                                           'ubiquitous, the visual attention '
                                           'histories of readers in the 719\n'
                                           'same reading states indeed differ '
                                           'from each other, which can be '
                                           'found from Observation II, '
                                           'Observation III, 720\n'
                                           'Observation IV, Observation V, and '
                                           'Observation VI. 721\n'
                                           'RQ2: When readers are in the same '
                                           'reading states, e.g., encountering '
                                           'difficulty progressing, how does '
                                           'reader visual 722\n'
                                           'attention interact with semantic '
                                           'cues in the text? 723\n'
                                           'Conforming hypothesis 2: When '
                                           'readers encounter the same '
                                           'processing difficulties, they '
                                           'shift their visual 724\n'
                                           'attention to the surrounding text '
                                           'to fetch contextual semantic cues. '
                                           'In other words, when readers’ '
                                           'reading 725\n'
                                           'progress is blocked, easy text '
                                           'that is semantically related to '
                                           'complex text also receives more '
                                           'visual attention and 726\n'
                                           'cognitive effort, which can be '
                                           'found from Observation II and '
                                           'Observation III. 727\n'
                                           '6.5.2 Discussion and Future Work. '
                                           '728\n'
                                           '(1) Science of reading. This work '
                                           'investigates the human cognitive '
                                           'reading process by exploring the '
                                           'com- 729\n'
                                           'plementarity of eye movements and '
                                           'text. However, it is also '
                                           'important to integrate '
                                           'illustration information to 730\n'
                                           'understand how people read. A '
                                           'recent study has shown that '
                                           'text-diagram instructions can '
                                           'improve reading 731\n'
                                           'comprehension [ 37]. Thus, our '
                                           'future work aims to exploit '
                                           'semantic information, including '
                                           'text and illustrations, 732\n'
                                           'and integrate them with eye '
                                           'movements. In addition, we aim to '
                                           'investigate more reading states '
                                           'that might provide 733\n'
                                           'a complete picture of the reading '
                                           'cognitive progress. In addition to '
                                           'determining the reading states at '
                                           'the word 734\n'
                                           'and sentence levels, it would be '
                                           'valuable to measure how people '
                                           'read at the entire passage level. '
                                           'This could 735\n'
                                           'deepen our understanding of how '
                                           'people summarize and reflect on '
                                           'learned knowledge during reading. '
                                           '736\n'
                                           '(2) Interactive Reading System. '
                                           'Our system is still an early-stage '
                                           'prototype. A longer user study '
                                           'would 737\n'
                                           'enable the collection of more data '
                                           'and user feedback to improve the '
                                           'interactive design and user '
                                           'experience. 738\n'
                                           'Proc. ACM Interact. Mob. Wearable '
                                           'Ubiquitous Technol., Vol. 0, No. '
                                           '0, Article 0. Publication date: '
                                           '2022.CASES: A Cognition-Aware '
                                           'Smart Eyewear System for '
                                           'Understanding How People Read '
                                           '•0:27\n'
                                           'This could help us to build a '
                                           'mature reading assistance system '
                                           'that contributes to educational '
                                           'applications, HCI 739\n'
                                           'studies, etc. 740\n'
                                           '(3) Reading Contexts. We would '
                                           'like to emphasise that we presume '
                                           'that the system will be '
                                           'well-migrated 741\n'
                                           'to various reading scenarios, and '
                                           'therefore, we use the eyeglasses '
                                           'form to study reading. We believe '
                                           'wearing 742\n'
                                           'eyeglasses to read is a portable '
                                           'way in numerous reading contexts, '
                                           'including computerized reading and '
                                           'physical 743\n'
                                           'reading (e.g., reading '
                                           'newspapers). However, since our '
                                           'eyeglasses are still in the early '
                                           'prototype stage, in this 744\n'
                                           'work, we did not experimentally '
                                           'cover all the scenarios. The '
                                           'system presented in this work is '
                                           'currently used in 745\n'
                                           'a computerized-reading context, as '
                                           'reading using electronic devices '
                                           'has become common in our modern '
                                           'lives 746\n'
                                           'and has been widely studied by a '
                                           'large body of researchers [ '
                                           '13,30,54]. We are aware that '
                                           'investigating the 747\n'
                                           'physical-reading context is also '
                                           'important, and we are interested '
                                           'in applying our eyeglasses to '
                                           'investigate the 748\n'
                                           'reading (cognitive) states under '
                                           'this context in our future work. '
                                           '749\n'
                                           '(4) Brain-Sensing Methods in '
                                           'Reading. In addition to '
                                           'eye-tracking in reading, '
                                           'brain-sensing via electroen- 750\n'
                                           'cephalograph (EEG) can determine '
                                           'the level of cognitive workload '
                                           'under different rapid serial '
                                           'visual presentation 751\n'
                                           'settings, as demonstrated in [ '
                                           '45]. It can be utilized to '
                                           'determine the cognitive workload '
                                           'or attention of texts at 752\n'
                                           'different granularity levels. '
                                           'However, this has to be done with '
                                           'the eye movement data jointly to '
                                           'accurately locate 753\n'
                                           'the positions of text being read '
                                           'and allow fine-grained analysis on '
                                           'processing difficulty of words. We '
                                           'believe that 754\n'
                                           'it is a direction that is worth '
                                           'exploring in the future to further '
                                           'improve the performance of our '
                                           'system. 755\n'
                                           '\n',
   'Introduction': ' 25\n'
                   'Reading is a fundamental approach to learning, through '
                   'which people can expand their vocabulary, gain knowl- 26\n'
                   'edge, and develop skills. Research has shown a positive '
                   'relationship between reading and learning; for example, '
                   '27\n'
                   'the more people read, the more effectively they improve '
                   'vocabulary, knowledge levels, and cognitive skills [ 15]. '
                   '28\n'
                   'In fact, reading has long been considered the most '
                   'important path to lifelong learning, and lifelong readers '
                   'are 29\n'
                   'generally more successful, both personally and '
                   'professionally [24, 76]. 30\n'
                   'The science of reading has attracted decades of interest '
                   'in human-computer interaction (HCI) [ 27,81], cognitive '
                   '31\n'
                   'science [ 40,44], psychology [ 67], educational psychology '
                   '[ 11,77], cognition and neuroscience [ 82], pedagogy [ '
                   '38], 32\n'
                   'Author’s address: Anonymous Authors.\n'
                   'Permission to make digital or hard copies of all or part '
                   'of this work for personal or classroom use is granted '
                   'without fee provided that\n'
                   'copies are not made or distributed for profit or '
                   'commercial advantage and that copies bear this notice and '
                   'the full citation on the first\n'
                   'page. Copyrights for components of this work owned by '
                   'others than ACM must be honored. Abstracting with credit '
                   'is permitted. To copy\n'
                   'otherwise, or republish, to post on servers or to '
                   'redistribute to lists, requires prior specific permission '
                   'and/or a fee. Request permissions from\n'
                   'permissions@acm.org.\n'
                   '©2022 Association for Computing Machinery.\n'
                   '2474-9567/2022/0-ART0 $15.00\n'
                   'https://doi.org/00.0000/00.00000\n'
                   'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., '
                   'Vol. 0, No. 0, Article 0. Publication date: 2022.0:2 '
                   '•Anonymous Authors\n'
                   'and brain science [ 2,51]. Reading is a cognitive process '
                   'and understanding it benefits numerous research commu- 33\n'
                   'nities. Studying how people understand the semantics and '
                   'syntax of text can aid in understanding natural language '
                   '34\n'
                   'representation and processing, which are key '
                   'functionalities of human-level intelligence [ 35]. '
                   'Understanding 35\n'
                   'the reading process can also advance the theory of human '
                   'behavior, thus benefiting the domains of applied 36\n'
                   'psychology, pedagogy, and educational psychology. For '
                   'instance, we can scrutinize human cognitive abilities [ '
                   '36] 37\n'
                   'such as verbal working, memory capacity, inhibitory '
                   'control ability, perceptual speed, and immediate and '
                   'delayed 38\n'
                   'effects on reading processes. Furthermore, understanding '
                   'how people read sheds light on reading patterns and 39\n'
                   'strategies, potentially helping readers achieve '
                   'metacognitive awareness and read more efficiently [ '
                   '21,58,84]. 40\n'
                   'In particular, HCI researchers have studied enhancing '
                   'human reading efficiency [ 80], reading proficiency [ 52], '
                   '41\n'
                   'reading skills [55], reading comprehension performance '
                   '[34, 43], and reading outcomes [28]. 42\n'
                   'Reading is a multi-level interactive eye-mind cognitive '
                   'process. In the short term, readers visually perceive 43\n'
                   'each word, encode it, and mentally assign semantics. In '
                   'the long term, readers visually perceive a sentence and '
                   '44\n'
                   'mentally associate it with context and domain knowledge [ '
                   '39]. Reading can be viewed as a sequence of numerous 45\n'
                   'time-varying states. For instance, some studies explored '
                   'the state of mind wandering, to detect whether a reader '
                   '46\n'
                   'is cognitively engaged or decoupled from the current '
                   'reading task [ 19,54]. Furthermore, some researchers '
                   'studied 47\n'
                   'the state of having difficulty processing unfamiliar words '
                   '[ 33,72]. However, we note that processing difficulties '
                   '48\n'
                   'can present at multiple granularities, e.g., readers may '
                   'encounter difficulties at the level of a single word, a '
                   '49\n'
                   'sentence, or a paragraph. Since it is hard to enumerate '
                   'all reading states, we focus on the problem of probing 50\n'
                   'the reading cognitive process to detect and explain '
                   'multiple states at word and sentence levels. Specifically, '
                   '51\n'
                   'we investigate whether a reader’s mind is wandering, '
                   'whether the reader is positively engaged, and when 52\n'
                   'comprehension is delayed due to word- or sentence-level '
                   'processing difficulties. 53\n'
                   'Eye movements are good indicators to infer the cognitive '
                   'process [ 1,64,74,83]. This is based on the eye- 54\n'
                   'mind hypothesis [ 39], which states that there is a close '
                   'relationship between where the eyes look and where 55\n'
                   'the mind is engaged. Owing to the fast development of '
                   'eye-tracking technologies, we can easily access eye- 56\n'
                   'tracking data [ 3,50] to explore eye-mind relationships. '
                   'Numerous researchers have extended the relationship 57\n'
                   'between eye movements and cognitive processes [ 65,67]. '
                   'Also, numerous prevalent methods design eye-tracking 58\n'
                   'reading systems to automatically track the participants’ '
                   'eye movements in a non-intrusive way [ 16,38,72,73]. 59\n'
                   'These works have summarized some hand-engineered eye '
                   'movement features to probe the reading cognitive 60\n'
                   'process [72, 73]. 61\n'
                   'However, eye-tracking technologies suffer from a number of '
                   'shortcomings. The error of commercially 62\n'
                   'available eye-tracking technologies typically ranges from '
                   '1 to 4 degrees [ 46,60]. Under reading scenarios, this 63\n'
                   'angular accuracy translates to a spatial tracking '
                   'resolution of about 1.4–2.6 cm. Considering a '
                   'computerized- 64\n'
                   'reading task where the distance from eye to screen is '
                   '40–50 cm, this means that the resolution of the eye '
                   'tracker is 65\n'
                   'about 3 to 4 lines for a single-spaced document and about '
                   '1 to 3 words in the horizontal direction. Such low spatial '
                   '66\n'
                   'resolution makes it infeasible to track reading states '
                   'during word-by-word and line-by-line reading because we '
                   '67\n'
                   'cannot locate the words and lines accurately. Previous '
                   'studies tackle this problem by using an unrealistic '
                   'setting 68\n'
                   'with a very wide line spacing (e.g., triple-spaced [ 16]), '
                   'leaving them unsuitable for use with normally spaced text. '
                   '69\n'
                   'In addition, eye-tracking techniques are subject to the '
                   'inherent transient jitter [ 9] of human gaze and vertical '
                   '70\n'
                   'drift, which require constant calibration [ 7]. '
                   'Eye-tracking techniques suited to real-world scenarios '
                   'have the 71\n'
                   'potential to advance the study of reading. 72\n'
                   'Furthermore, existing methods ignore contextual influences '
                   'from text, resulting in less accurate 73\n'
                   'reading state estimation and undermining semantic '
                   'explanation for these states. Given the same reading 74\n'
                   'context and motivations, the factors influencing reading '
                   'states mainly pertain to the reading material’s and 75\n'
                   'subject’s domain knowledge about the content. For example, '
                   'a good reader may cross-reference previously read 76\n'
                   'text to assist in understanding new and unfamiliar text [ '
                   '33]. In such cases, the high reading frequencies of the '
                   '77\n'
                   'earlier text do not necessarily imply that they are '
                   'difficult. To correctly estimate the current reading '
                   'state, it 78\n'
                   'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., '
                   'Vol. 0, No. 0, Article 0. Publication date: 2022.CASES: A '
                   'Cognition-Aware Smart Eyewear System for Understanding How '
                   'People Read •0:3\n'
                   ' InterventionEstimation and semantic explaination        '
                   'Gaze point\n'
                   'DNNEye\n'
                   'videoScene\n'
                   'video\n'
                   'Prevalent work in this field\n'
                   'primarily focuses on leveraging\n'
                   'eye gaze patterns to reveal the\n'
                   'while-reading cognitive processes.\n'
                   'However , eye gaze patterns suffer\n'
                   'from limited resolution, jitter noise,\n'
                   'and cognitive biases, resulting in\n'
                   'limited accurac y in tracking\n'
                   'cognitive reading states.\n'
                   'While-reading\n'
                   'Prevalent work in this\n'
                   'field primarily focuses\n'
                   'on leveraging eye gaze\n'
                   'patterns to reveal the \n'
                   'cognitive processes. popularpopular\n'
                   'Scene camera \n'
                   '1. Estimation:\n'
                   '    processing dif ficulty at     \n'
                   '    word "Prevalent"\n'
                   '2. Semantic explanation: \n'
                   '    challenging word, long     \n'
                   '    gazing duration, ......。。。Prevalent work in this\n'
                   'field primarily focuses\n'
                   'on leveraging eye gaze\n'
                   'patterns to reveal the \n'
                   'cognitive processes. \n'
                   '。。。\n'
                   '。。。Prevalent work in this\n'
                   'field primarily focuses\n'
                   'on leveraging eye gaze\n'
                   'patterns to reveal the \n'
                   'cognitive processes. Eye camera \n'
                   'Fig. 1. The proposed CASES smart eyewear system.\n'
                   'is important to be aware of the semantic meaning of the '
                   'current text, the cross-referenced text, their semantic '
                   '79\n'
                   'correlations, and real-time eye gaze patterns. However, it '
                   'is a non-trivial task to properly fuse the semantics of '
                   '80\n'
                   'reading text and eye movements and learn from them in '
                   'progressive reading scenarios, and it is more challenging '
                   '81\n'
                   'to infer semantic explanations for reading state '
                   'timeseries. 82\n'
                   'This work aims to provide accurate estimations and '
                   'semantic explanations for reading state timeseries to 83\n'
                   'support research and outreach efforts in the field of '
                   'reading science. To this end, we pose the following two '
                   '84\n'
                   'research questions (RQ) and posit the corresponding '
                   'hypotheses. 85\n'
                   'RQ1: Do readers in the same reading states show different '
                   'visual attention distributions on the reading text? 86\n'
                   'Hypothesis 1: Readers in the same reading state will show '
                   'varying visual attention histories (detailed in 87\n'
                   'Section 3), e.g., different total fixation duration, '
                   'reading times, scanning paths, etc. That is, the visual '
                   'attention 88\n'
                   'histories of readers in the same reading state differ from '
                   'each other. 89\n'
                   'RQ2: When readers are in the same reading states, e.g., '
                   'encountering difficulty progressing, how does reader 90\n'
                   'visual attention interact with semantic cues in the text? '
                   '91\n'
                   'Hypothesis 2: As indicated by previous studies [ 19,73], '
                   'readers’ cognitive effort in processing text is positively '
                   '92\n'
                   'related to the difficulty of the text. However, in '
                   'contrast with previous studies, we further hypothesize '
                   'that 93\n'
                   'readers can overcome reading difficulties by fetching '
                   'contextual semantic cues from the surrounding text. When '
                   '94\n'
                   'progress is blocked, easy text that is semantically '
                   'related to difficult text also receives more visual '
                   'attention and 95\n'
                   'cognitive effort. 96\n'
                   'The motivation for this work is that the semantic context '
                   'of text has a direct impact on the multi-level interactive '
                   '97\n'
                   'eye-brain cognitive reading process. Leveraging the rich '
                   'semantic information about reading materials, which 98\n'
                   'can be extracted by advanced natural language processing '
                   '(NLP) techniques [ 61,87], can improve estimation 99\n'
                   'accuracy and provide semantic interpretation of reading '
                   'states. The semantic information is high-resolution 100\n'
                   'because NLP models can provide semantics at the word level '
                   '[ 61,87]. The inherent hierarchical structure of 101\n'
                   'the semantic information can also be inferred by '
                   'summarizing the semantics of words to a sentence level. '
                   'The 102\n'
                   'high-resolution semantic information can compensate for '
                   'the low-resolution eye movements for more accurate 103\n'
                   'reading state tracking. More importantly, the real-time '
                   'interaction of eye movements and semantic context can 104\n'
                   'provide semantic explanations for the ongoing reading '
                   'states. 105\n'
                   'To this end, we present a Cognition-Aware Smart Eyewear '
                   'System (CASES) capable of measuring reading 106\n'
                   '(cognitive) state timeseries. Figure 1 illustrates the '
                   'workflow of the proposed system. At the heart of CASES is '
                   'a 107\n'
                   'bi-modal multi-task network named CASES-Net, which takes '
                   'the bi-modal data, i.e., the eye-tracking and reading 108\n'
                   'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., '
                   'Vol. 0, No. 0, Article 0. Publication date: 2022.0:4 '
                   '•Anonymous Authors\n'
                   'text data, as inputs and estimates cognitive reading '
                   'states in real-time at two granularities: word and '
                   'sentence 109\n'
                   'level. To collect high-quality bi-modal data, CASES uses '
                   'two cameras to record the two required modalities 110\n'
                   'automatically: an outward-facing scene camera to capture '
                   'text and an inward-facing camera to track gaze points 111\n'
                   'during reading. CASES is implemented in the form of '
                   'eyewear to avoid interfering with the reading process 112\n'
                   'when collecting data. Surveys are deferred until after a '
                   'reading task is completed, also to avoid interference. '
                   '113\n'
                   'CASES-Net employs a four-layer temporal convolutional '
                   'network (TCN) based module to fuse the two 114\n'
                   'types of sequential modalities, one of which is informed '
                   'by semantic information extracted from a pre-trained 115\n'
                   'NLP model [ 6,18]. We treat estimations at two '
                   'granularities as two distinct but related tasks and '
                   'propose a 116\n'
                   'shared convolutional filter mechanism within the TCN to '
                   'learn the characteristics of the two tasks and their 117\n'
                   'commonalities. Moreover, we design a multi-task and '
                   'hierarchical loss function to guide reading state '
                   'estimation. 118\n'
                   'To evaluate CASES, we first collect and construct a '
                   'dataset and then demonstrate that CASES has higher reading '
                   '119\n'
                   'state estimation accuracy than baseline methods. To sum '
                   'up, CASES-Net combines gaze and semantic information 120\n'
                   'to better estimate reading states. More importantly, it '
                   'provides semantic explanations for these reading states. '
                   '121\n'
                   'The well-trained deep model can automatically detect when '
                   'users encounter reading difficulty without requiring 122\n'
                   'further inputs (e.g., feedback) from them, thereby '
                   'limiting potential subjective biases. 123\n'
                   'This work makes the following contributions. 124\n'
                   '•We present a cognition-aware smart eyewear system (CASES) '
                   'to probe and explain human cognitive 125\n'
                   'processes while reading. CASES aims to support the study '
                   'of reading and learning to read, as well as 126\n'
                   'supporting HCI and educational applications investigations '
                   'on improving reading productivity. The CASES 127\n'
                   'system is equipped with a deep neural network, CASES-Net, '
                   'that extracts features pertaining to the visual 128\n'
                   'attention history and text semantic content. It fuses the '
                   'two types of features via a shared convolutional 129\n'
                   'filter mechanism based on TCN to enable accurate reading '
                   'state estimation at various granularities. 130\n'
                   '•CASES is evaluated in real-world contexts. We conduct an '
                   'ablation study involving 25 participants, in 131\n'
                   'which CASES delivered superior reading state detection to '
                   'baseline methods. Specifically, encoding text 132\n'
                   'semantic content facilitates learning from context cues '
                   'and improves reading state estimation accuracy. 133\n'
                   'Compared with the conventional eye-tracking-only method, '
                   'we improve accuracy by 20.90% for sentence. 134\n'
                   'Furthermore, the text semantic context enables '
                   'quantitative explanations of reading (cognitive) states. '
                   '135\n'
                   '•We integrate CASES into a novel interactive reading '
                   'assistant system. Three and a half months of deployment '
                   '136\n'
                   'with 13 in-field studies demonstrate that the integrated '
                   'system can enable helpful interventions for readers, 137\n'
                   'thus improving self-awareness in the reading process and '
                   'helping readers adopt more effective reading 138\n'
                   'habits. 139\n'
                   'The rest of this paper is organized as follows. Section 2 '
                   'surveys related work. Section 3 clarifies the key concepts '
                   '140\n'
                   'used in this work. Section 4 details the proposed network '
                   'and our built real-time reading state detection and 141\n'
                   'intervention system. Section 5 presents the experimental '
                   'setups and results. Section 6 presents our findings when '
                   '142\n'
                   'using CASES in practice, general discussion, and future '
                   'direction. Finally, Section 7 concludes this work. 143\n'
                   '\n',
   'Method': ' 202\n'
             'This section clarifies three important concepts used in this '
             'work: eye movements, visual attention, and semantic 203\n'
             'attention. 204\n'
             'Eye Movements: Eye movement patterns can reveal reading '
             'strategies and are vital to understanding the reading 205\n'
             'cognitive process. As shown in existing studies [ 16,53], '
             'reading generally consists of a series of pauses and rapid 206\n'
             'shifts in gaze locations. The pauses are called fixation and the '
             'shifts are called saccades. These patterns reflect 207\n'
             'the low-level oculomotor characteristics during reading, '
             'typically determined by the physical properties of text, 208\n'
             'such as the positions or lengths of words. 209\n'
             'By exploring eye movement patterns, researchers establish '
             'connections between low-level eye movement 210\n'
             'behaviors and higher-level cognitive processes during reading [ '
             '74]. First, research shows that the direction and 211\n'
             'duration of eye fixation reveal how the cognitive process '
             'unfolds over time [ 72,73]. More specifically, fixation 212\n'
             'locations indicate the attended content, while fixation duration '
             'suggests the level of cognitive effort invested by 213\n'
             'the reader, i.e., longer fixation suggests more effort. Second, '
             'the processing time-course of eye movement patterns 214\n'
             'is widely used to reveal the temporally continuous reading '
             'process, which is often linked with comprehending or 215\n'
             'memorizing. For example, one common temporal reading activity is '
             'to move the gaze backward to review the 216\n'
             'already-read content. In this case, the informative eye movement '
             'patterns might be the reading and regression 217\n'
             'durations, which is also called the second pass [ 31]. Finally, '
             'to alleviate the potential inter-person variations, recent 218\n'
             'work also designs global features or statistical features based '
             'on eye movement patterns to access the reading 219\n'
             'process, such as the number of saccades, saccade frequencies, '
             'and variations in fixation duration [ 16]. Given 220\n'
             'the potential ability of eye movement patterns in revealing '
             'reading cognitive processes, this work also employs 221\n'
             'these hand-engineered features as valuable indicators. However, '
             'to better suit our case, we first distinguish the 222\n'
             'representing eye movement patterns at two granularities, and '
             'then we re-design them at word and sentence 223\n'
             'levels, respectively. More details can be found in Section '
             '4.1.3. 224\n'
             'Visual Attention: Although no previous work explicitly defines '
             'visual attention in reading scenarios, substantial 225\n'
             'studies demonstrate a strong correlation between eye movement '
             'patterns and attentional processing during 226\n'
             'reading. For instance, the E-Z reader model [ 66] posits that '
             'attention during reading moves from word to 227\n'
             'word continuously. The serial-processing assumption states that '
             'attention is linked to changes of focus in text 228\n'
             'processing [ 26,56,85]. Following these studies, our work '
             'describes visual attention during reading by establishing 229\n'
             'the connection between eye movement patterns and the '
             'corresponding while-reading text components, such as 230\n'
             'words and sentences. More specifically, we define the visual '
             'attention state to be the collection of eye movement 231\n'
             'features on each text component. For example, when reading the '
             'sentence “ They race to maturity, with the shortest 232\n'
             'generation time of any vertebrate ”, the visual attention for '
             'the word “ vertebrate ” consists of fixation duration, 233\n'
             'reading times, number of fixations, etc. At the sentence level, '
             'the visual attention state is defined based on the 234\n'
             'total dwell time, saccade times, etc. 235\n'
             'Semantic Attention: We are interested in exploring how the '
             'semantic meaning from text assists in estimating the 236\n'
             'time-series reading states and how they explain these states. '
             'From this perspective, it is necessary to have a holistic 237\n'
             'semantic understanding of while-reading texts. Furthermore, such '
             'understanding should cover the semantic 238\n'
             'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., Vol. 0, '
             'No. 0, Article 0. Publication date: 2022.CASES: A '
             'Cognition-Aware Smart Eyewear System for Understanding How '
             'People Read •0:7\n'
             'meaning of different grain sizes of texts, ranging from single '
             'words and sentences to passage levels. This works 239\n'
             'terms this semantics collection at various granularities as '
             'semantic attention . For example, semantic attention can 240\n'
             'hint at whether the while-reading text components are difficult. '
             'These difficult components may be unfamiliar or 241\n'
             'ambiguous words or sentences with complex syntax, which often '
             'delay reading. In this case, appropriately using 242\n'
             'such semantic meaning regarding the difficult score can provide '
             'additional evidence in revealing the current 243\n'
             'reading state and deliver a reasonable interpretation regarding '
             'why the current text components block the 244\n'
             'reading. More details regarding semantic attention can be found '
             'in Section 4.1.2. 245\n'
             '\n'
             ' 246\n'
             'This section describes the CASES system design. We first detail '
             'the CASES network (CASES-Net), a deep neural 247\n'
             'network for detecting and interpreting ongoing reading states. '
             'Then, we describe a real-time reading state 248\n'
             'estimation and intervention system aiming to boost reading '
             'comprehension performance. 249\n'
             'SAE\n'
             '......Time\n'
             '......VAE......\n'
             'Eye cameraScene camera123Word\n'
             'Eye\n'
             'tracking\n'
             '1\n'
             'T......TimeConcatenatedPooling\n'
             'Reading state estimationTemporal\n'
             'convolutional network...\n'
             'gaze point............"liberation" "the" "hardly"......\n'
             '......"… is pleasent but hardly the liberation that …"\n'
             '...... ......"… is pleasent but hardly the liberation that …"\n'
             'Fig. 2. Framework of the CASES.\n'
             '4.1 CASES Network 250\n'
             '4.1.1 Overall Pipeline. Figure 2 depicts the general framework '
             'of CASES-Net. It consists of four modules: semantic 251\n'
             'attention extraction (SAE), visual attention extraction (VAE), '
             'cross-attention extraction (CAE), and reading state 252\n'
             'estimation/explanation. 253\n'
             'The first step in the CASES pipeline provides a comprehensive '
             'semantic understanding of the text before the 254\n'
             'reading begins. This semantic meaning information compensates '
             'for the low-resolution eye-tracking data, thus 255\n'
             'enabling accurate reading state estimation. Semantic meaning '
             'also enables explanations during reading state 256\n'
             'detection tasks in later pipeline stages. To extract semantic '
             'meaning, the system turns on the outward-facing 257\n'
             'scene camera to obtain the text to be read. The SAE module then '
             'runs once on the text. It utilizes NLP techniques 258\n'
             'to extract the high-resolution semantic features and the '
             'inherent linguistic structure from the text, thus facilitating '
             '259\n'
             'subsequent tasks. 260\n'
             'Texts contain rich semantic information, but for better '
             'individual reading state estimation, personalized visual 261\n'
             'attention data are also necessary. To capture it, the VAE module '
             'is triggered to obtain the online visual attention 262\n'
             'features corresponding with text components (e.g., while-gazing '
             'words or sentences). More specifically, the 263\n'
             'CASES system senses reader eye images to predict gaze sequences '
             'using continuous eye-tracking [ 46,60]. Then, 264\n'
             'the VAE module extracts visual attention features from the '
             'sequential gaze data. In parallel, the scene camera 265\n'
             'records time-aligned scene images to help track gaze positions. '
             '266\n'
             'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., Vol. 0, '
             'No. 0, Article 0. Publication date: 2022.0:8 •Anonymous Authors\n'
             'Since the obtained semantic meaning of the text and visual '
             'attention features are at different spatial resolutions, 267\n'
             'we propose the CAE module to properly align them. Here, we use '
             'words to segment the visual attention features, 268\n'
             'because words are the minimal units considered in this work, '
             'upon which sentences and global context depend. 269\n'
             'The TCN-based network estimates the reading states at word and '
             'sentence levels, aiming to explore the task- 270\n'
             'specific features for the assistance of the multi-task output. '
             'One feature represents the binary determination of 271\n'
             'whether a reader has difficulty processing a word; we call this '
             'the “word-level task”. The second task is hierarchical 272\n'
             'multi-label classification at the sentence level, which includes '
             '(Task I) estimating whether a reader is having 273\n'
             'sentence-level processing difficulty and if so, (Task II) '
             'estimating whether the reader is facing comprehension 274\n'
             'challenges, the reader’s mind is wandering, or both. A '
             'multi-task and hierarchical loss function for training 275\n'
             'guides CASES-Net. We can qualitatively understand the reasons '
             'for the predicted reading states by visualizing 276\n'
             'the learned semantic attention and visual attention features. '
             '277\n'
             'The rest of this section explains the technical details of each '
             'of the proposed modules. 278\n'
             '4.1.2 Semantic Attention Extraction Module. The aim of the SAE '
             'module is to understand the high-resolution 279\n'
             'semantic meaning of the document R, ranging from the word level '
             'to the document level. There are two primary 280\n'
             'prerequisites for extracting accurate semantic features: '
             'obtaining the while-gazing locations and text contents. 281\n'
             'The former, i.e., while-gazing locations, can be obtained by '
             'using eye tracking and represented as Points of Gaze 282\n'
             '(PoG) timeseries. Each PoG corresponds to a two-dimensional '
             'coordinate in the scene image recorded by the 283\n'
             'scene camera. Given the locations of PoG, we can easily load the '
             'while-gazing text contents because the reading 284\n'
             'system has already stored all the reading materials in advance. '
             'After that, we propose to extract the following 285\n'
             'three types of semantic features by utilizing various advanced '
             'pre-trained NLP models. 286\n'
             '(1)Each word in Ris encoded as a 768-dimensional vector by XLNet '
             'model [ 86], which can learn the semantic 287\n'
             'meaning of the document by processing the whole text passage '
             'once. To lower the potential adverse effect 288\n'
             'incurred by the high dimensionality, we reduce the XLNet '
             'features to 64 dimensions via a fully-connected 289\n'
             '(FC) layer and denote them as r𝐵={r𝐵\n'
             '𝑤}𝑊\n'
             '𝑤=1, where r𝐵\n'
             '𝑤∈R64and𝑊is the total number of words. 290\n'
             '(2)To understand the keyword information in the document, we '
             'calculate the probability of each word 291\n'
             'describing the whole document via the YAKE model [ 6]. The '
             'keyword features are denoted as r𝐾={𝑟𝐾\n'
             '𝑤}𝑊\n'
             '𝑤=1, 292\n'
             'where 𝑟𝐾\n'
             '𝑤∈R. 293\n'
             '(3)We use word difficulty to assist in the final task of '
             'identifying the reading state. Following Franklin et 294\n'
             'al. [22], we describe the word difficulty using the length of '
             'the word, number of syllables, and familiarity 295\n'
             'scored by the MRC psycholinguistic database [ 14]. We denote the '
             'difficulty of words by r𝐷={r𝐷\n'
             '𝑤}𝑊\n'
             '𝑤=1, 296\n'
             'where r𝐷\n'
             '𝑤=[𝑙𝑤, 𝑠𝑤, 𝑓𝑤]∈R3. 297\n'
             'Finally, each word in the document is represented by the '
             'concatenation of the three feature vectors; that is 298\n'
             'r𝑤=[r𝐵\n'
             '𝑤, 𝑟𝐾\n'
             '𝑤,r𝐷\n'
             '𝑤]∈R68(𝑤=1,2, . . . ,𝑊 ). Note that the semantics regarding more '
             'coarse levels (e.g., sentence- and 299\n'
             'passage- level) can be generalized from that of the word level, '
             'as words are inherently structured and semantically 300\n'
             'connected — a passage consists of multiple sentences and a '
             'sentence of multiple words. 301\n'
             '4.1.3 Visual Attention Extraction Module. A reliable gaze '
             'sequence is the foundation for accurate visual attention 302\n'
             'feature extraction. However, the raw gaze points are noisy due '
             'to difficult-to-avoid human motion and limited 303\n'
             'eye-tracking resolution. To alleviate this issue, we design a '
             'filtering algorithm to smooth the raw gaze points, 304\n'
             'leveraging their sequential characteristics. More specifically, '
             'we first employ an existing eye-tracking technology 305\n'
             'to estimate the PoGs and record the PoGs sequences as E={e𝑡}𝑇\n'
             '𝑡=1, where 𝑇is the total number of timestamps 306\n'
             'considered. The designed filtering method first uses median '
             'filtering to discard outliers due to gaze jitter. Then, 307\n'
             'we use mean filtering to stabilize the fluctuations of '
             'sequential PoGs due to the limited eye-tracking resolution. 308\n'
             'After filtering, we obtain the smoothed PoGs E∗={e∗\n'
             '𝑡}𝑇\n'
             '𝑡=1. We segment each word and sentence using E∗and 309\n'
             'then send them to the next step for visual attention extraction. '
             '310\n'
             'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., Vol. 0, '
             'No. 0, Article 0. Publication date: 2022.CASES: A '
             'Cognition-Aware Smart Eyewear System for Understanding How '
             'People Read •0:9\n'
             'The number of PoGs will increase rapidly during reading. To '
             'reduce the size of PoGs, experts have engineered 311\n'
             'a large number of representative features reflecting how people '
             'comprehend characters during reading [ 16,31, 312\n'
             '72,73] or whether they are disengaged from reading [ 19,54]. In '
             'this work, we propose to further enrich the 313\n'
             'engineered visual features. The following features are widely '
             'used to describe word-level processing state while 314\n'
             'reading: fixation duration, number of fixations, and number of '
             'repeated word readings. However, we observe that 315\n'
             'these three features vary not only person-to-person but also '
             'during reading. Such variation significantly affects 316\n'
             'estimation performance. The personal variation is usually '
             'removed by normalizing personal data [ 29]; however, 317\n'
             'the latter while-reading variation is rarely considered. This '
             'work introduces local information to tackle the latter 318\n'
             'problem: every 𝜏seconds, we add the statistical features to '
             'describe the mean and the variance of each engineered 319\n'
             'feature, for E∗={e∗\n'
             '𝑡}𝜏\n'
             '𝑡=1, to describe the visual attention for each word. In total, '
             'we obtain a 9-dimensional feature 320\n'
             'for each word. Moreover, we normalize the four sentence-level '
             'representative visual features, including dwell 321\n'
             'time [ 17], saccade times [ 20], forward saccade times [ 59], '
             'and backward saccade times [ 59], using the sentence 322\n'
             'length, so these features better describe the local variation. '
             'Given that we segment 𝑀words during 𝜏, each word 323\n'
             'is represented using r𝐸\n'
             '𝑤∈R(9+4)(𝑤=1,2, . . . , 𝑀 ). There are nine word-level features '
             'and four sentence-level 324\n'
             'features that are identical to the words in the same sentence. '
             '325\n'
             'Lastly, we propose to use the deep features of the sequential '
             'gaze data, as recent studies have demonstrated 326\n'
             'the effectiveness of deep neural networks (DNN) on eye movement '
             'classification. We adopt the existing feature 327\n'
             'extractor based on the 1D-CNN with BLSTM backbone [ 75] (denoted '
             'asN𝑒𝑦𝑒to extract 8-dimensional deep 328\n'
             'features during time duration 𝜏, i.e., r𝐷𝑁𝑁\n'
             '𝑤∈R8(𝑤=1,2, . . . , 𝑀 ), and discard the classifier. 329\n'
             '4.1.4 Cross-Attention Extraction Module. To facilitate '
             'downstream multi-task learning, the cross-attention 330\n'
             'extraction (CAE) module first fuses the two modalities, then '
             'explores the commonalities to predict at different 331\n'
             'granularities and the distinct task-specific information. 332\n'
             'Before fusing the two modalities, we use the following strategy '
             'to time synchronize them. Specifically, for 333\n'
             'each smoothed PoGs sequence e∗\n'
             '𝑡, we identify the 𝑀words being processed at time 𝑡, and '
             'concatenate the two 334\n'
             'features vectors to obtain f𝑤\n'
             '𝑡=[r𝑤,r𝐸\n'
             '𝑤,r𝐷𝑁𝑁\n'
             '𝑤]∈R(68+13+8)as the overall representation of the two '
             'modalities. 335\n'
             'For all other words 𝑤′that have not been visually processed till '
             'time 𝑡, we pad the semantic attention feature 336\n'
             'vector r𝑤′with a zero vector, i.e., f𝑤′\n'
             '𝑡=[r𝑤′∈R68,0∈R21]. In this way, the word being processed at time '
             '𝑡can 337\n'
             'be properly described semantically with its corresponding visual '
             'attention features. In contrast, the unread words 338\n'
             'padded with zeros are given less attention. 339\n'
             'The CAE module uses a Temporal Convolutional Network (TCN) '
             'model, which can capture temporal de- 340\n'
             'pendencies. Specifically, the module uses temporal convolutional '
             'filters/kernels to process input sequences. 341\n'
             'Each filter calculates a weighted average in the time domain, '
             'and the parameters of the filters are learned to 342\n'
             'optimize the objective function. Each TCN layer consists of '
             'temporal convolutions, a non-linear Relu activation 343\n'
             'function, and a max pooling function or an upsampling function. '
             'The CAE module has four TCN layers. To learn 344\n'
             'different tasks more efficiently, the filters of the last layer '
             'are divided into task-specific filters, namely word-level 345\n'
             'filters/sentence-level filters, and task-shared filters, namely '
             'common filters. The features extracted by word-level 346\n'
             'filters and common filters are used for word-level tasks. The '
             'features extracted by sentence-level filters and 347\n'
             'common filters are used for sentence-level tasks. 348\n'
             '4.1.5 Reading States Estimation and Explanations. After '
             'obtaining the cross-attention features, we are ready to 349\n'
             'detect the reading state of “processing difficulty”. We have the '
             'following three tasks. (1) Word-level binary-class 350\n'
             'classification task 𝑇𝑤𝑜𝑟𝑑: The word-level features are fed to a '
             'fully connected layer to predict whether the reader 351\n'
             'finds the word being processed difficult. Sentence-level and '
             'word-level tasks differ. Since we know that mind 352\n'
             'wandering may co-occur with reading difficulty for a sentence, '
             'we formulate the task at the sentence level in the 353\n'
             'following hierarchical fashion. (2) Sentence-level binary-class '
             'classification task 𝑇𝑠𝑒𝑛𝑡, 1: With the sentence-level 354\n'
             'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., Vol. 0, '
             'No. 0, Article 0. Publication date: 2022.0:10 •Anonymous '
             'Authors\n'
             'features, we first determine whether the reader is in a normal '
             'reading state without any processing difficulties 355\n'
             'using a binary classifier. (3) Sentence-level multi-label '
             'classification task 𝑇𝑠𝑒𝑛𝑡, 2: If the reader enters into an 356\n'
             'abnormal state, the reader can be either mind wandering or '
             'processing difficulty, or both; This is a multi-label 357\n'
             'classification task, where multi labels can be assigned '
             'simultaneously; label 1 is mind wandering and label 2 is 358\n'
             'processing difficulty. 359\n'
             'Finally, to train the network, we propose the following loss '
             'function reflecting the performances of all tasks: 360\n'
             'L=L(𝑇𝑤𝑜𝑟𝑑)+L( 𝑇𝑠𝑒𝑛𝑡, 1)+L( 𝑇𝑠𝑒𝑛𝑡, 2). (1)\n'
             'Binary Cross Entropy (BCE) loss is used for 𝑇𝑤𝑜𝑟𝑑andL(𝑇𝑤𝑜𝑟𝑑)is '
             'illustrated as follows\n'
             'L(𝑇𝑤𝑜𝑟𝑑)=−1\n'
             '𝑊𝑊∑︁\n'
             '𝑤=1\x12\n'
             '𝑦𝑤𝑜𝑟𝑑\n'
             '𝑤 log𝑝𝑤𝑜𝑟𝑑\n'
             '𝑤+(1−𝑦𝑤𝑜𝑟𝑑\n'
             '𝑤)log(1−𝑝𝑤𝑜𝑟𝑑\n'
             '𝑤)\x13\n'
             ',\n'
             'where 𝑊denotes the number of word; 𝑦𝑤𝑜𝑟𝑑\n'
             '𝑤 denotes the label of word 𝑤,𝑦𝑤𝑜𝑟𝑑\n'
             '𝑤=0indicates the reader finds\n'
             'the word 𝑤easy, 𝑦𝑤𝑜𝑟𝑑\n'
             '𝑤=1indicates the reader finds the word 𝑤difficult; 𝑝𝑤𝑜𝑟𝑑\n'
             '𝑤 is the word-level estimation\n'
             'results given by the network N𝑤𝑜𝑟𝑑.\n'
             'BCE loss is also used for 𝑇𝑠𝑒𝑛𝑡, 1andL(𝑇𝑠𝑒𝑛𝑡, 1)is illustrated '
             'as follows\n'
             'L(𝑇𝑠𝑒𝑛𝑡, 1)=−1\n'
             '𝑆𝑆∑︁\n'
             '𝑠=1\x12\n'
             '𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 log𝑝𝑠𝑒𝑛𝑡, 1\n'
             '𝑠+(1−𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠)log(1−𝑝𝑠𝑒𝑛𝑡, 1\n'
             '𝑠)\x13\n'
             ',\n'
             'where 𝑆denotes the number of sentences; 𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 denotes the binary classification label of the 𝑠th sentence,\n'
             '𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 =0indicates the reader is in a normal reading state without '
             'any processing difficulties for sentence 𝑠,\n'
             '𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 =1indicates the reader is in an abnormal reading state; 𝑝𝑠𝑒𝑛𝑡, '
             '1\n'
             '𝑠 is the sentence-level binary classification\n'
             'estimation results given by the network N𝑠𝑒𝑛𝑡, 1.\n'
             'For sentences with 𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 =1, to solve the multi-label problem, BCE loss is used for '
             'each label separately and the\n'
             'loss of 𝑇𝑠𝑒𝑛𝑡, 2is illustrated as follows\n'
             'L(𝑇𝑠𝑒𝑛𝑡, 2)=−1\n'
             'Í𝑆\n'
             '𝑠=11(𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 =1)𝑆∑︁\n'
             '𝑠=1𝐿∑︁\n'
             '𝑙=11(𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 =1)\x12\n'
             '𝑦𝑠𝑒𝑛𝑡, 2\n'
             '𝑠,𝑙log𝑝𝑠𝑒𝑛𝑡, 2\n'
             '𝑠,𝑙+(1−𝑦𝑠𝑒𝑛𝑡, 2\n'
             '𝑠,𝑙)log(1−𝑝𝑠𝑒𝑛𝑡, 2\n'
             '𝑠,𝑙)\x13\n'
             ',\n'
             'where 𝐿=2denotes the number of labels, i.e. label 1 as mind '
             'wandering and label 2 as processing difficulty; 361\n'
             '𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠,𝑙denotes the supervised information of the 𝑙th label for '
             'sentence 𝑠,𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠,𝑙=1indicates sentence 𝑠has the 362\n'
             '𝑙th label, 𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠,𝑙=0indicates sentence 𝑠does not have the 𝑙th label; 1(·)is an '
             'indicator function, 1(𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 =1)=1 363\n'
             'when 𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 =1,1(𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 =1)=0when 𝑦𝑠𝑒𝑛𝑡, 1\n'
             '𝑠 =0;𝑝𝑠𝑒𝑛𝑡, 2\n'
             '𝑠,𝑙is the sentence-level multi-label estimation results 364\n'
             'given by the network N𝑠𝑒𝑛𝑡, 2. 365\n'
             '4.2 EYEReader: A Real-Time Reading State Detection and '
             'Intervention System 366\n'
             'Our goal is to determine reading state series that influence '
             'reading fluency and mitigate the negative effects of 367\n'
             'reading processing difficulties. To this end, we build a '
             'real-time reading state detection and intervention system 368\n'
             '(called EYEReader) for English language. For the convenience of '
             'readers, EYEReader is implemented in the form 369\n'
             'of a website, enabling cross-platform compatibility. 370\n'
             'This section first gives a concrete example to show the key '
             'features of EYEReader and how to use it. Then it 371\n'
             'details the system architecture, along with its operation '
             'pipeline. At last, it describes the hardware prototype. 372\n'
             'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., Vol. 0, '
             'No. 0, Article 0. Publication date: 2022.CASES: A '
             'Cognition-Aware Smart Eyewear System for Understanding How '
             'People Read •0:11\n'
             'equal\n'
             '(a)\n'
             'Imagine they think dif ferently than us, and\n'
             'need more sensory input or time to process it\n'
             'before reaching a mental state. (b)\n'
             'You just mind wandered;\n'
             'please reread the missed content. (c)\n'
             'Fig. 3. Screenshots of three intervention examples. Detection '
             'and Interventions: at the word level (left), (b) at the '
             'sentence\n'
             'level (middle), and (c) on mind wandering (right).\n'
             '4.2.1 Key Features and Operation Process of EYEReader. We first '
             'give some key features of EYEReader, and we 373\n'
             'then use a concrete case to show the automatic detection and '
             'intervention process. 374\n'
             'Key feature 1: text materials selection. The text materials '
             'should contain various topics, as the intervention 375\n'
             'is anticipated to be text-agnostic. We select 36 reading '
             'comprehension materials with diverse topics from an 376\n'
             'English qualification test to match the participants’ reading '
             'comprehension ability. Each article has around 450 377\n'
             'words on average. Users can log in to the system, select their '
             'preferred articles from existing materials, and start 378\n'
             'reading by simply clicking a button. 379\n'
             'Key feature 2: friendly reading interface. Because we have '
             'overcome the limited resolution issues when 380\n'
             'eye-tracking is used during reading scenarios, the interface of '
             'text presentation of EYEReader is similar to 381\n'
             'common computerized reading settings. More specifically, '
             'articles are automatically divided into several different 382\n'
             'pages (around 240 words per page) with a regular line height, '
             'approximately single-spaced. We adopt an 18-point 383\n'
             'default font typeface. 384\n'
             'Key feature 3: intervention design. The interventions are '
             'designed to help users overcome the three 385\n'
             'while-reading processing difficulties, i.e. mind wandering, '
             'challenging words and complex sentences, that may 386\n'
             'lead to a negative impact on their reading comprehension '
             'performance. Three interventions are designed for 387\n'
             'the three difficulties respectively: 1) providing an immediate '
             'reminder once mind wandering is detected, which 388\n'
             'reminds readers to focus on the current reading; 2) simplifying '
             'the challenging words; and 3) streamlining the 389\n'
             'complex sentences. We provide the following three examples to '
             'further clarify how the interventions support 390\n'
             'reading. 391\n'
             '(1)Simplifying challenges words. Once the system detects that a '
             'reader is facing a challenging word, it 392\n'
             'highlights the word in blue and provides a more comprehensible '
             'one in the pop-up window. For example, 393\n'
             'when a user struggles with “coextensive”, the pop-up window '
             'offers a more straightforward and easy- 394\n'
             'to-understand one, “equal”. Figure 3 (a) provides a screenshot '
             'of this intervention. After receiving the 395\n'
             'interventions, users can click on the highlighted words to hide '
             'the reminders and continue reading. 396\n'
             '(2)Streamlining complex sentences. The procedure of streamlining '
             'complex sentences is similar to that of 397\n'
             'simplifying challenging words. Differently, the system '
             'highlights the complex sentences in red and provides 398\n'
             'simpler sentences in the pop-up window. For example, for a long '
             'and complex sentence, “Imagine that they 399\n'
             'not only come to the belief in a different way than we, but that '
             'the sensory stimulations that suffice for us 400\n'
             'do not suffice for most of them, at least without lengthy '
             'calculation, they do not go immediately from the 401\n'
             'sensory stimulations to the “mental state”.” the system provides '
             'a relatively more straightforward version: 402\n'
             '“Imagine they think differently than us, and need more sensory '
             'input or time to process it before reaching 403\n'
             'a mental state.” A screenshot of this type of intervention is '
             'depicted in Figure 3 (b). Users can also click on 404\n'
             'the highlighted sentences to hide the pop-up window and continue '
             'reading. 405\n'
             'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., Vol. 0, '
             'No. 0, Article 0. Publication date: 2022.0:12 •Anonymous '
             'Authors\n'
             '(3)Giving mind wandering reminder. When the system detects that '
             'the reader is distracted while reading, it 406\n'
             'highlights the missed content in yellow and displays a pop-up '
             'message in the centre of the screen, showing 407\n'
             'that “You just mind wandered; please reread the missed content.” '
             'A screenshot of this type of intervention 408\n'
             'is depicted in Figure 3 (c). The pop-up message automatically '
             'fades out after one second. 409\n'
             'Participants’ eye gazes are calibrated prior to their reading in '
             'order to correlate the two cameras equipped in 410\n'
             'the eyewear. The calibration method follows Pupil Capture1[41]. '
             'Specifically, during the calibration phase, the 411\n'
             'participants wear the eyewear and sit in front of the computer, '
             'a pupil calibration marker appears on the screen 412\n'
             'with fixed locations. The participant is instructed to gaze at '
             'the maker for approximately two seconds. The same 413\n'
             'procedure is executed for the other four calibration markers on '
             'the screen. In this way, the system would record 414\n'
             'these positions to correlate the two cameras. 415\n'
             'During the reading process, readers wear the prototype eyeglass '
             'and sit in front of the computer to read. The 416\n'
             'pre-trained CASES-Net model is always-on to automatically detect '
             'potential abnormal reading states, i.e., whether 417\n'
             'the user is struggling with difficult words or complex '
             'sentences, or their mind is wandering. When abnormal 418\n'
             'events that affect reading are detected, the system triggers '
             'interventions automatically. The text components will 419\n'
             'be highlighted, and the corresponding treatments will be shown '
             'on the right-top of the text content automatically 420\n'
             'in a pop-up window. 421\n'
             'Database Web server\n'
             'User\n'
             'Eyewear3. Eye-tracking data, extracted visual attention features '
             'and interventions4. Historical eye-tracking data, visual '
             'attention features and text data 5. Interventions\n'
             '6. Interventions\n'
             '2. Gaze data\n'
             'Reading texts\n'
             'Visual attention features\n'
             'Interventions User Id\n'
             'Gaze dataDatabase collection\n'
             '1. Eye images and scene images\n'
             'PC/Laptop\n'
             'Fig. 4. The architecture of the reading state detection and '
             'intervention system.\n'
             '4.2.2 System Architecture. Figure 4 illustrates the overall '
             'architecture of EYEReader. We use the Vue.js framework 422\n'
             'to develop the front-end website, while we choose Django for the '
             'back-end of the website, as it is a widely-used 423\n'
             'Python web framework [ 5]. Django offers a variety of '
             'third-party tools for building communication between the 424\n'
             'front-end and back-end efficiently following the REST API '
             'specification. To store and manage the data on the 425\n'
             'server, we adopt one of the widely-used open-source database '
             'management systems — MySQL [ 57]. The eyewear 426\n'
             '1https://docs.pupil-labs.com/core/software/pupil-capture/#calibration\n'
             'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., Vol. 0, '
             'No. 0, Article 0. Publication date: 2022.CASES: A '
             'Cognition-Aware Smart Eyewear System for Understanding How '
             'People Read •0:13\n'
             'and the PC used for reading are in the same LAN(Local Area '
             'Network). The eyewear is running on Android 427\n'
             '12. We developed a service app without user interfaces to read '
             'the real-time video stream recorded by the two 428\n'
             'cameras and push the video stream to the PC via the RTSP '
             'protocol2. On the PC edge, we receive the coming 429\n'
             'video stream from the eyewear using the RTSP protocol. The '
             'received video is then handled by Pupil Capture 430\n'
             'and Pupil Service provided by Pupil Labs3. 431\n'
             'Next, we describe the overall operation workflow of the built '
             'intervention system and show how it provides 432\n'
             'just-in-time interventions for users encountering reading '
             'processing difficulties. There are mainly six steps 433\n'
             'described below. 434\n'
             '•Step 1: During system operation, EYEReader loads the '
             'pre-trained CASES-Net from the server when 435\n'
             'receiving the requests from the front end. 436\n'
             '•Step 2: The recorded eye/scene images captured by eyewear are '
             'pushed to the user’s PC for eye-tracking 437\n'
             'using the Pupil Capture [41]. 438\n'
             '•Step 3: The tracked gaze points are sent to the server for '
             'further visual attention feature extraction. 439\n'
             '•Step 4: The server loads the historical eye-tracking data, '
             'visual attention features, and texts to decide 440\n'
             'whether it is the right time to intervene. 441\n'
             '•Step 5: Once processing difficulties are detected, the '
             'estimation results are returned to the front end for 442\n'
             'triggering interventions. The corresponding treatment is shown '
             'at the front end to facilitate the current 443\n'
             'reading. 444\n'
             '•Step 6: After that, the current interventions and all other '
             'data are saved in Database. 445\n'
             '(a) 3D eyeglass frame (b) Front view (c) Lateral view12\n'
             'Scene cameraEye camera\n'
             'Fig. 5. Hardware prototype of CASES eyewear.\n'
             '4.2.3 Hardware Design. We design prototype eyewear and integrate '
             'CASES-Net into the eyewear, as eyewear is 446\n'
             'a natural way to be used in various reading scenarios. 447\n'
             'We presume that the eyewear will be well-migrated to various '
             'reading scenarios. Therefore, we adopt a 448\n'
             'stand-alone scheme to integrate the computing components and '
             'power supply into the headset frame. Figure 5 449\n'
             'shows the eyewear hardware prototype. 450\n'
             'The eye-tracker follows the Pupil3, and we make slight '
             'adjustments to suit our case. More specifically, we use 451\n'
             'Qualcomm Snapdragon 865 platform directly integrated into the '
             'left leg of the eyewear. The eye camera and 452\n'
             'scene camera modules are replaced with 20 MegaPixels (MP) '
             'Samsung S5K3T2 and 64 MP Samsung S5KGW1, 453\n'
             '2https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol\n'
             '3https://docs.pupil-labs.com/core/diy/\n'
             'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., Vol. 0, '
             'No. 0, Article 0. Publication date: 2022.0:14 •Anonymous '
             'Authors\n'
             'respectively. The eye camera is used to record eye videos to '
             'perform eye tracking. The scene camera senses scene 454\n'
             'videos to capture the text being read. We design the 3D eyeglass '
             'frame to fit the two cameras into the left leg 455\n'
             'of the mounting frame. To balance the weight of the headset, the '
             'battery is integrated into the right leg of the 456\n'
             'eyewear. 457\n'
             '\n',
   'Related Work': ' 144\n'
                   'This work is mostly relevant to three broad areas: reading '
                   'science, eye-tracking in reading, and natural language '
                   '145\n'
                   'processing. 146\n'
                   '2.1 Science of Reading 147\n'
                   'Reading science has attracted decades of interest in '
                   'various research communities, e.g., HCI, pedagogy, and '
                   '148\n'
                   'educational psychology. These studies primarily deal with '
                   'the outcomes of reading [ 28] and reading compre- 149\n'
                   'hension [ 43]. Recently, researchers have studied reading '
                   'patterns and strategies that improve the efficiency of '
                   '150\n'
                   'reading [ 21,27,58,84], e.g., interactive reading systems '
                   'that detect mind wandering during reading [ 19,54]. 151\n'
                   'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., '
                   'Vol. 0, No. 0, Article 0. Publication date: 2022.CASES: A '
                   'Cognition-Aware Smart Eyewear System for Understanding How '
                   'People Read •0:5\n'
                   'They mitigated the negative effect of mind wandering on '
                   'reading comprehension using just-in-time interven- 152\n'
                   'tions [ 19,54]. Other methods detect words readers do not '
                   'know automatically [ 27] and provide appropriate 153\n'
                   'help [ 33,72]. In psychology, applied psychology, and '
                   'educational psychology, researchers primarily focused 154\n'
                   'on studying how texts are read and comprehended [ '
                   '11,62,67,77,83]. For example, Perfetti et al. delivered a '
                   '155\n'
                   'blueprint of reading, consisting of the visual process, '
                   'representation process that converts visual perception '
                   'into 156\n'
                   'a linguistic representation, and operation process on the '
                   'representation [ 62]. In cognition science, neuroscience, '
                   '157\n'
                   'and brain science, extensive reading studies focus on '
                   'developing computational theories of cognition [ 47]. One '
                   '158\n'
                   'important branch studies the representations and '
                   'processing of natural languages by the human brain [ 35]. '
                   'For 159\n'
                   'example, Lewis et al. contributed a theoretical framework '
                   'to explain how verbal working memory supports 160\n'
                   'sentence processing [ 47]. Kamide et al. studied how the '
                   'global and local information in texts impact sentence 161\n'
                   'processing [ 40]. Cognitive scientists usually jointly '
                   'consider language representation and processing [ 23] '
                   'based 162\n'
                   'on the belief that discovering language representation can '
                   'help answer questions about computation, and vice 163\n'
                   'versa. Schrimpf et al. provided computationally explicit '
                   'evidence that language comprehension mechanisms in 164\n'
                   'human brains are fundamentally shaped by predictive '
                   'processing through an integrative modeling approach [ 69]. '
                   '165\n'
                   'In summary, previous works on the science of reading '
                   'primarily focus on leveraging eye-tracking during 166\n'
                   'reading to study the reading process and outcomes. '
                   'However, they focus less on how individual readers '
                   'perceive 167\n'
                   'and process the text in real-time. This study introduces '
                   'context information from texts to the study of reading '
                   '168\n'
                   'cognitive processes. 169\n'
                   '2.2 Eye-Tracking in Reading 170\n'
                   'Eye-tracking technology can acquire real-time eye '
                   'movements in a non-intrusive manner [ 8]. It is natural to '
                   '171\n'
                   'utilize eye movement data to probe the reading process, as '
                   'the reading process initiates visual input and operates '
                   '172\n'
                   'as an interactive eye-mind cognition process [ 39]. Over '
                   'the past decades, numerous studies have focused on 173\n'
                   'analyzing eye movement data obtained during reading to '
                   'understand the reading cognitive process and provide 174\n'
                   'reading assistance [ 4,19,25,32,54,73]. For example, '
                   'Hyrskykari proposed a gaze-aware reading assistance system '
                   '175\n'
                   'to provide help at the right time without interrupting the '
                   'reader’s thoughts [ 32]. Cheng et al. proposed a social '
                   '176\n'
                   'reading system, in which they demonstrated that sharing '
                   'eye gaze annotations generated by experts promoted 177\n'
                   'reading comprehension for non-experts [ 10]. Bottos and '
                   'Balasingam presented an approach to accurately track 178\n'
                   'the horizontal eye-gaze points in reading scenarios [ 4]. '
                   'In addition, there are also many studies focused on 179\n'
                   'detecting reading behaviors, such as mind wandering [ '
                   '19,54] or encountering difficulties in comprehending 180\n'
                   'unfamiliar words [33, 72]. 181\n'
                   'In general, these relevant methods have demonstrated that '
                   'eye movement data helps understand the reading 182\n'
                   'cognitive process. However, the semantic information of '
                   'the text, which is closely related to the reading process, '
                   '183\n'
                   'is rarely used in previous studies. This study jointly '
                   'considers text semantic information and eye movement data '
                   '184\n'
                   'can facilitate understanding the reading process and how '
                   'readers comprehend texts. 185\n'
                   '2.3 Nature Language Processing 186\n'
                   'Natural language processing (NLP) uses computational '
                   'techniques to represent and analyze human languages [ 12] '
                   '187\n'
                   '(see [ 42] for a comprehensive review). NLP can usually be '
                   'classified into two categories: natural language under- '
                   '188\n'
                   'standing and natural language generation. As discussed '
                   'above, this work uses natural language understanding 189\n'
                   'techniques to obtain semantic contextual information from '
                   'texts. Successful natural language understanding 190\n'
                   'techniques can provide generic models for NLP downstream '
                   'tasks, such as analyzing the association among 191\n'
                   'text components [ 18], extracting keywords [ 6,71], and '
                   'analyzing syntax [ 48]. For example, Linzen et al. pointed '
                   '192\n'
                   'out that, given targeted syntax supervision, a long '
                   'short-term memory (LSTM) network can learn syntax infor- '
                   '193\n'
                   'mation [ 49]. Later, they further stated that linguists '
                   'and neural network researchers might contribute to each '
                   '194\n'
                   'Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., '
                   'Vol. 0, No. 0, Article 0. Publication date: 2022.0:6 '
                   '•Anonymous Authors\n'
                   'other’s areas [ 48]. Furthermore, NLP neural networks can '
                   'provide good representations of text; for example, the '
                   '195\n'
                   'bidirectional encoder representations from transformers '
                   '(BERT) model [ 18], which is based on transformers [ 79], '
                   '196\n'
                   'can obtain state-of-the-art results on several NLP tasks '
                   'by providing high-quality language representations. 197\n'
                   'Considering the dependency between the masked positions '
                   'and the discrepancy from pretrain-finetune that 198\n'
                   'BERT neglects, Yang et al. proposed a generalized '
                   'autoregressive pretraining method to overcome the '
                   'limitations 199\n'
                   'of BERT [ 86]. Their pre-trained model, XLNet, outperforms '
                   'BERT on various tasks. This work builds on recent 200\n'
                   'progress in NLP by using pre-trained NLP models to help '
                   'understand the reading cognitive process. 201\n'
                   '\n'})]
    sections = related_works[0][0]
    paper = related_works[0]
    field = 'Computer Science'
    topic = 'Use both the semantic context of text and visual attention during reading to more accurately predict the ' \
            'temporal sequence of cognitive states'
    ref_path = '/Users/luqi/PycharmProjects/writing/users/luqi-0123/references/Introduction'
    section = 'Introduction'
    outline = {'outline': {'Discussion': {'points': ['Interpretation of the experimental results',
                                          'Comparison with existing methods and their limitations',
                                          'Implications of accurately predicting cognitive states during reading',
                                          'Future directions and potential improvements']},
                           'Experiment and Experiment and Result': {'points': ['Details of the evaluation methodology',
                                                     'Selection of text materials for the experiment',
                                                     'Creation of the dataset for training and testing',
                                                     'Analysis of the overall performance of the proposed approach',
                                                     'Feedback and insights gathered from the participants']},
                           'Introduction': {'points': ['Background on the importance of understanding cognitive states during reading',
                                            'Shortcomings of existing methods in predicting cognitive states',
                                            'Overview of the proposed approach using semantic context and visual attention']},
                           'Method': {'points': ['Explanation of eye movements and their relation to cognitive processes',
                                      'Analysis of visual attention and eye movement patterns',
                                      'Understanding semantic attention and its role in comprehending texts',
                                      'Description of the proposed network architecture for predicting reading states']},
                           'Related Work': {'points': ['Previous research on reading comprehension and cognitive states',
                                            'Existing techniques for eye-tracking in reading',
                                            'Relevant studies in natural language processing'
                                            ]}
                           },
               'title': 'Predicting the Temporal Sequence of Cognitive States using Semantic Context and Visual Attention during Reading'}
    draft = ['阅读不仅是看字识词，还包括理解句子和联系所学知识。在阅读时，我们可能会走神、集中注意力，或者在理解某些单词或句子时遇到难题。这项研究主要是想弄清楚，当我们阅读时，这些不同的情况是怎样发生的。',
             '使用眼动追踪技术检测阅读时的认知状态准确度不够高，很难精确知道你在看哪个单词或哪行文字。这个技术在一般阅读距离下，最多只能大概知道你看的是哪几行或哪几个单词。而且，由于人眼自然会有些微小的抖动和移动，眼动追踪设备需要经常调整。所以，在普通的阅读研究中，用这种技术有些困难。此外，现有的阅读时认知状态检测方法没怎么考虑文本本身的内容，这让判断不太准确，也不容易解释为什么会出现这样的状态。比如，懂得更多的人可能利用以前的知识来理解新内容，但这并不意味着之前的内容很难。要准确判断阅读状态，需要了解你正在读的内容和你眼睛的动作。但把这两者结合起来，特别是解释阅读状态，是个挺难的事情。',
             '我们设计了一种叫CASES的智能眼镜系统，用于实时测量阅读状态。这个系统利用眼动追踪和文本数据来估计人们在阅读时的单词和句子层面的状态。CASES通过两个摄像头收集数据，减少对阅读的干扰，并使用先进的分析技术来处理这些数据。测试结果表明，CASES在估计阅读状态的准确性方面优于其他方法，并能提供状态的语义解释，同时减少了对用户反馈的依赖。'
             ]
    revise_text_test1(field, topic, section, outline, draft,
                      ref_path, related_works)
