import os
import streamlit as st

from writing import print_outline, outline_to_dict, string_to_markdown, revise_text_test, save_pdf_chroma, \
    generate_outline_test1, get_related_work_outline, revise_text_test1

from tools import init_dirs, change_outline_dict, writing_print_text, save_pdf, split_pdf_by_outline

from pprint import pprint
import shutil


# 设置页面为宽屏模式
st.set_page_config(layout="wide")

# 使用 HTML 使标题居中
st.markdown("<h1 style='text-align: center;'>AI Scientific Research Paper Writing Assistant 💡</h1>",
            unsafe_allow_html=True)

# # 使用 columns 创建两个列
# col1, col2 = st.columns([1, 1], gap="large")  # 创建两个等宽的列

# 生成内容的状态管理
if 'generated_outline' not in st.session_state:
    st.session_state['generated_outline'] = None

if 'generated_text' not in st.session_state:
    st.session_state['generated_text'] = None

# 检查是否已点击提交按钮
if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False

if 'display_text' not in st.session_state:
    st.session_state['display_text'] = None

if 'writing_sec' not in st.session_state:
    st.session_state['writing_sec'] = None

if 'writing_text' not in st.session_state:
    st.session_state['writing_text'] = None

if 'all_text' not in st.session_state:
    st.session_state['all_text'] = None

if 'topic' not in st.session_state:
    st.session_state['topic'] = None

if 'field' not in st.session_state:
    st.session_state['field'] = None

if 'markdown_text' not in st.session_state:
    st.session_state['markdown_text'] = None


path_prefix = os.getcwd()
username = ''
field = ''
topic = ''
related_works = []

if 'submitted' not in st.session_state or st.session_state['submitted'] == False:
    with st.form("user_info"):
        username = st.text_input('Please input your username')
        research_fields = ['Physics', 'Mathematics', 'Quantitative Biology', 'Computer Science', 'Quantitative Finance',
                                  'Statistics', 'Electrical Engineering and Systems Science', 'Economics']
        field = st.selectbox('Please select the research field', research_fields)
        st.session_state['field'] = field
        # 创建一个文本输入框供用户输入研究主题
        topic = st.text_input('Please input the research topic')
        st.session_state['topic'] = topic

        outline = st.session_state.get('generated_outline', '')

        uploaded_files = st.file_uploader("Please upload the template papers in PDF format", accept_multiple_files=True, type=['pdf'])

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            if topic:
                st.session_state['submitted'] = True
                path_prefix = os.path.join(path_prefix, username)
                path = os.path.join(path_prefix, 'related_works')
                init_dirs(path_prefix)
                shutil.rmtree(path)
                for file in uploaded_files:
                    save_pdf(file, path)
                related_works = split_pdf_by_outline(path)
                works_tmp = []
                rel_outline = ''

                for work in related_works:
                    print(work[1])
                    out, wo = get_related_work_outline(work)
                    rel_outline += out
                    rel_outline += '\n\n'
                    works_tmp.append(wo)
                related_works = works_tmp
                st.session_state['generated_outline'] = print_outline(generate_outline_test1(field, topic, rel_outline))
                outline = st.session_state['generated_outline']
                st.rerun()


if 'submitted' in st.session_state and st.session_state['submitted']:
    # 使用 columns 创建两个列
    col1, col2 = st.columns([1, 1], gap="large")  # 创建两个等宽的列
    with col1:
        tab1, tab2 = st.tabs(["Outline", "Writing"])
        with tab1:
            with st.container():
                st.info('The outline has been generated based on the research topic and the template papers\n'
                        '- You can manually modify in the following area\n'
                        '- Click the "Confirm" button to save the outline and proceed to writing', icon="ℹ️")

            outline = st.text_area(label='Outline', key='Outline', value=st.session_state['generated_outline'], height=220)

            # 显示生成的文本
            if 'generated_outline' in st.session_state and st.session_state['generated_outline']:
                if st.button('Confirm', key='accept_outline'):
                    if not st.session_state['all_text']:
                        st.session_state['all_text'] = outline_to_dict(outline)
                    else:
                        st.session_state['all_text'] = change_outline_dict(outline, st.session_state['all_text'])
                    # 在右侧显示文本
                    st.session_state['display_text'] = writing_print_text(st.session_state['all_text'])
                    st.rerun()

        with tab2:
            def reset_text(display, default):
                if 'sec' in st.session_state:
                    writing_sec = st.session_state['sec']
                else:
                    writing_sec = default
                st.session_state['generated_text'] = display['outline'][writing_sec]['content']
                st.session_state['writing_text'] = display['outline'][writing_sec]['draft']


            display = {}
            if 'display_text' in st.session_state and st.session_state['display_text']:
                path_prefix = os.path.join(path_prefix, username)
                if st.session_state['all_text']:
                    display = st.session_state['all_text']
                secs = [str(sec) for sec in display['outline']]
                section = st.selectbox('Please choose the section', secs, on_change=lambda: reset_text(display, secs[0]), key='sec')
                st.session_state['writing_sec'] = section
                points = display['outline'][section]['points']
                uploaded_files = st.file_uploader("Please upload the references in PDF format", accept_multiple_files=True,
                                                  type=['pdf'])
                draft = ['' for point in points]

                for i in range(len(points)):
                    draft[i] = st.text_area(label=points[i], key=points[i], value=display['outline'][section]['draft'][i], height=110)

                revision = st.text_area(label='Revision', key='revision', value=st.session_state['generated_text'], height=210)

                revise, accept = st.columns([1, 1])
                # splits = ''
                if revise.button('Revise'):
                    ref_path = os.path.join(path_prefix, 'references', section)
                    vec_path = os.path.join(path_prefix, 'related_works', 'chroma')
                    for file in uploaded_files:
                        save_pdf(file, ref_path)

                    st.session_state['generated_text'] = revise_text_test1(field, topic, section, display, draft,
                                                                          ref_path, related_works)
                    revision = st.session_state['generated_text']
                    st.rerun()
                if st.session_state['generated_text']:
                    if accept.button('Confirm', key='accept_text'):
                        # 在右侧显示文本
                        display['outline'][section]['content'] = revision
                        display['outline'][section]['draft'] = draft
                        st.session_state['all_text'] = display
                        st.session_state['display_text'] = writing_print_text(display)
                        st.rerun()

            else:
                st.error("You have not yet planned your writing outline. Please select 'Outline' in the menu bar to proceed with the plan.", icon="🚨")

    # 在右边的列（col2）中显示用户已接受的文本
    with col2:
        if 'prev_display_text' not in st.session_state:
            st.session_state['prev_display_text'] = None

        if 'display_text' in st.session_state and st.session_state['display_text'] and \
                st.session_state['prev_display_text'] != st.session_state['display_text']:
            st.session_state['markdown_text'] = string_to_markdown(st.session_state['display_text'])
            st.session_state['prev_display_text'] = st.session_state['display_text']
        container = st.container(border=True)
        if 'markdown_text' in st.session_state:
            container.markdown(st.session_state['markdown_text'])
