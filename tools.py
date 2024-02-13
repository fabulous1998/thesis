import os
import arxiv

import shutil
import subprocess
import PyPDF2
import pdfplumber
from pprint import pprint
import re


def search_arxiv(field, topic, num):
    """根据研究领域和主题搜索 arXiv 上的论文"""
    # 构建查询字符串，将研究领域和主题结合起来
    query = f"cat:{field} AND {topic}"

    search = arxiv.Search(
        query=query,
        max_results=num,
        sort_by=arxiv.SortCriterion.Relevance
    )
    infos = arxiv.Client().results(search)
    papers = [
        {
            "Arxiv": paper.entry_id,
            "Title": paper.title,
            "Authors": paper.authors,
            "Submitted Date": paper.published,
            "Abstract": paper.summary,
            "Comment": paper.comment,
            "Journal": paper.journal_ref,
            "Citation": ''
        }
        for paper in infos
    ]
    return papers


def paper_print_text(papers, ignore=None):
    info = ""
    for paper in papers:
        for attr in paper:
            if attr not in ignore:
                text = attr + ": " + str(paper[attr]) + "\n"
                info += text
        info += "\n\n"
    return info


def search_for_outline(field, topic, num):
    papers = search_arxiv(field, topic, num)
    return "\n\n".join(
        [f"Title: {paper['Title']}\nAbstract: {paper['Abstract']}\nOutline: " for paper in
         papers])


def search_for_info(field, topic, num):
    papers = search_arxiv(field, topic, num)
    info = paper_print_text(papers)
    return info


def save_text(text, topic, path):
    save_path = os.path.join(path, 'related_works')
    save_path = os.path.join(save_path, topic)
    save_path = save_path + ".txt"
    with open(save_path, "w") as f:
        f.write(text)
    return save_path


def save_pdf(pdf, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, pdf.name)
    with open(path, "wb") as w:
        w.write(pdf.read())


def init_dirs(path):
    os.makedirs(os.path.join(path, "related_works"), exist_ok=True)
    os.makedirs(os.path.join(path, "chroma"), exist_ok=True)
    os.makedirs(os.path.join(path, "history"), exist_ok=True)
    os.makedirs(os.path.join(path, "references"), exist_ok=True)
    path = os.path.join(path, "references")
    os.makedirs(os.path.join(path, "Introduction"), exist_ok=True)
    os.makedirs(os.path.join(path, "Related Work"), exist_ok=True)
    os.makedirs(os.path.join(path, "Method"), exist_ok=True)
    os.makedirs(os.path.join(path, "Experiment and Result"), exist_ok=True)
    os.makedirs(os.path.join(path, "Discussion"), exist_ok=True)
    os.makedirs(os.path.join(path, "Conclusion"), exist_ok=True)


def print_outline(outline):
    str = outline['title'] + '\n'
    for sec in outline['outline']:
        str += '\n'
        str += sec
        str += '\n'
        if type(outline['outline'][sec]) == list:
            for point in outline['outline'][sec]:
                str += '- '
                str += point
                str += '\n'
        else:
            for point in outline['outline'][sec]['points']:
                str += '- '
                str += point
                str += '\n'
    return str


def print_section_outline(outline, section):
    str = section + '\n'
    if type(outline['outline'][section]['points']) == list:
        for point in outline['outline'][section]['points']:
            str += '- '
            str += point
            str += '\n'
    return str


def outline_to_dict(outline):
    outline = outline.split('\n\n')
    dic = {'title': outline[0], 'outline': {}}
    for i in range(1, len(outline)):
        out = outline[i]
        out = out.split('\n')
        dic['outline'][out[0]] = {'draft': ['' for i in range(1, len(out))], 'content': '', 'points': []}
        for j in range(1, len(out)):
            if len(out[j]) > 0:
                dic['outline'][out[0]]['points'].append(out[j].split('- ')[1])
    return dic


def writing_print_text(text):
    str = text['title'] + '\n'
    for sec in text['outline']:
        str += '\n'
        str += sec
        str += '\n'
        if not text['outline'][sec]['content']:
            for point in text['outline'][sec]['points']:
                str += '- '
                str += point
                str += '\n'
        else:
            str += text['outline'][sec]['content']
            str += '\n'
    return str


def change_outline_dict(outline, all_text):
    outline = outline_to_dict(outline)
    for sec in outline['outline']:
        for ori in all_text['outline']:
            if sec == ori:
                outline['outline'][sec]['draft'] = ['' for point in outline['outline'][sec]['points']]
                outline['outline'][sec]['content'] = all_text['outline'][sec]['content']
                break
    return outline


def split_pdf_by_outline(path):
    # Open a PDF document.
    pdf_list = os.listdir(path)
    pdf_list = [pdf for pdf in pdf_list if pdf != 'chroma' and pdf != 'splits']
    # outdir_prefix = os.path.join(path, 'splits')
    # os.makedirs(outdir_prefix, exist_ok=True)
    related_works = []
    for pdfname in pdf_list:
        pdf_path = os.path.join(path, pdfname)
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            bookmarks = [(bookmark.title, reader.get_destination_page_number(bookmark)) for bookmark in reader._get_outline() if
                         isinstance(bookmark, PyPDF2.generic.Destination)]
            bookmark = []
            sections = {}
            for i, (title, start_page) in enumerate(bookmarks):
                # Determine the end page for each section
                end_page = bookmarks[i + 1][1] if i + 1 < len(bookmarks) else len(reader.pages) - 1

                text = ""
                for j in range(start_page, end_page + 1):
                    page = reader.pages[j]
                    text += page.extract_text().encode('utf-8', 'replace').decode('utf-8')
                text = title.join(re.split(title, text, flags=re.IGNORECASE)[1:])

                if i < len(bookmarks) - 1:
                    text = "".join(re.split(bookmarks[i + 1][0], text, flags=re.IGNORECASE)[0])

                if len(text) > 0 and 'Abstract' not in title and 'Conclusion' not in title and 'References' not in title and 'Acknowledgments' not in title:
                    sections[title] = text
                    bookmark.append(title)
            related_works.append((reader.metadata.title, bookmark, sections))
    return related_works



if __name__ == '__main__':
    path = '/Users/luqi/PycharmProjects/writing/users/luqi_0108/related_works'
    split_pdf_by_outline(path)
