### NOT USED ### 

import gradio as gr
import pandas as pd
import numpy as np
import time
orig_csv_file = None
df = None

current_row = 0
USERS = 'admin'

def load_csv():
    global df
    global current_row

    if df:
        df.to_csv(f'review_{USERS}_{time.time()}.csv', index=False)
    df = pd.read_csv(orig_csv_file, dtype={'id':int, 'hs': str, 'cs': str, 'topic': str, 'tone': str, 'isCSContextuallyRelevant': str, 'isToneMatch': str})
    
    current_row = 0
    row_dict = df.iloc[current_row].to_dict()
    return row_dict['id'], row_dict['hs'], row_dict['cs'], row_dict['topic'], row_dict['tone'], row_dict['isCSContextuallyRelevant'], row_dict['isToneMatch']

def annotate_row(passfail, query_revised, answer_revised):
    global df
    global current_row

    df.at[current_row, 'visited'] = 1
    df.at[current_row, 'passfail'] = passfail
    df.at[current_row, 'query_revised'] = query_revised
    df.at[current_row, 'answer_revised'] = answer_revised

    if current_row < len(df) - 1:
        current_row += 1
    else:
        y = df[~df.visited]
        current_row = y.index[0]
    df.to_csv(f'review_{USERS}.csv', index=False)
    
    row_dict = df.iloc[current_row].to_dict()
    return row_dict['id'], row_dict['hs'], row_dict['cs'], row_dict['topic'], row_dict['tone'], row_dict['isCSContextuallyRelevant'], row_dict['isToneMatch'], 'annotated_data.csv'

def navigate(direction):
    global current_row
    if direction == "Previous":
        current_row = max(0, current_row - 1)
    elif direction == "Next":
        current_row = min(len(df) - 1, current_row + 1)
    elif direction == "First Unlabeled":
        unlabeled_row = df[df['isCSContextuallyRelevant'].isna()].index.min()
        if not np.isnan(unlabeled_row):
            current_row = int(unlabeled_row)

    row_dict = df.iloc[current_row].to_dict()
    return row_dict['id'], row_dict['hs'], row_dict['cs'], row_dict['topic'], row_dict['tone'], row_dict['isCSContextuallyRelevant'], row_dict['isToneMatch']

with gr.Blocks(theme=gr.themes.Soft()) as annotator:
    gr.Markdown("## Data Annotation")
    
    with gr.Row():
        btn_load = gr.Button("Load CSV")

    with gr.Row():
        gr.Markdown("### Current Row")
        with gr.Row():
            btn_previous = gr.Button("Previous")
            btn_next = gr.Button("Next")
            # btn_first_unlabeled = gr.Button("First Unlabeled")

        with gr.Row():
            passfail = gr.Radio(["1", "0"], label="pass or fail?")
            btn_annotate = gr.Button("Annotate")

        with gr.Row():
            idx = gr.Number(label='Index')
            with gr.Column():
                query = gr.Textbox(label='query')
                answer = gr.Textbox(label='answer')
            with gr.Column():
                query_revised = gr.Textbox(label='query_revised')
                answer_revised = gr.Textbox(label='answer_revised')

        with gr.Row():
            filename = gr.Textbox(label='filename')
            span = gr.Textbox(label='span')

    # with gr.Row():
    #     gr.Markdown("### Annotated Data File Download")
    #     file_download = gr.File()

    btn_load.click(load_csv, 
                   outputs=[idx, query, answer, filename, span, passfail])
    btn_annotate.click(annotate_row, inputs=[passfail, query_revised, answer_revised], outputs=[idx])
    btn_previous.click(navigate, inputs=gr.Textbox("Previous", visible=False), outputs=[idx])
    btn_next.click(navigate, inputs=gr.Textbox("Next", visible=False), outputs=[idx])

annotator.launch(share=True)