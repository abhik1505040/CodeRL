import plotly
import plotly.graph_objects as go
from transformers import AutoTokenizer
import glob
import json
import os
import numpy as np
from tqdm import tqdm
from datasets.utils import reindent_code

def frequency_graph(scores, x_axis, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, x_axis.replace(" ", "_") + ".html")

    print("*" * 100)
    print(
        x_axis + ">", 
        "max:", np.max(scores), 
        "min:", np.min(scores), 
        "avg:", np.mean(scores),
        "samples over max_seq_length:", sum(k > 512 for k in scores)
    )

    layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Histogram(
            histfunc='count',
            x=scores,
            xbins=dict(
                start=0,
                end=2000,
                size=1
            )
        )
    )

    fig.update_xaxes(
        title_text="<i>{}</i>".format(x_axis), 
        mirror=True, showline=True,
        linecolor='black', color='black', 
        showgrid=True, gridcolor='lightgray', gridwidth=1
    )

    fig.update_yaxes(
        title_text=f'<i>Frequencies</i>',
        mirror=True, showline=True,
        linecolor='black', color='black',
        showgrid=True, gridcolor='lightgray', gridwidth=1
    )
     
    fig.update_layout(
        showlegend=False, 
        font=dict(family='Latin Modern Math', size=15, color='black'), 
        legend=dict(x=0.8, y=1.0,traceorder='normal',font=dict(color='black', size=14), bordercolor='black', borderwidth=1)
    )

    plotly.offline.plot(fig, filename=output_path)


def main(root_dir, plot_dir, tokenizer_path='Salesforce/codet5-base'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    problem_dirs = sorted(glob.glob(root_dir + "/*"))
    
    input_lengths = []
    output_lengths = []

    for problem_name in tqdm(problem_dirs):
        question_fname = os.path.join(problem_name, "question.txt")
        sols_fname = os.path.join(problem_name, "solutions.json")            
        if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
            continue
            
        with open(question_fname, 'r') as f:
            question_str = f.read()
        with open(sols_fname) as f:
            solutions = json.load(f)

        starter_code = os.path.join(problem_name, "starter_code.py")    
        if (os.path.isfile(starter_code)):
            answer_type = "\nUse Call-Based format\n"
            with open(starter_code, 'r') as f:
                starter_code = f.read()
        else:
            answer_type = "\nUse Standard Input format\n"
            starter_code = ""

        input_str =  "\nQUESTION:\n" + question_str + "\n" + starter_code + "\n" + answer_type + "\nANSWER:\n"
        input_lengths.append(len(tokenizer.tokenize(input_str)))

        for solution in solutions:
            sol = reindent_code(solution)
            output_lengths.append(len(tokenizer.tokenize(sol)))

    del tokenizer
    frequency_graph(input_lengths, "Tokenized input lengths", plot_dir)
    frequency_graph(output_lengths, "Tokenized output lengths", plot_dir)

        
if __name__ == "__main__":
    main("data/APPS/train", "plots")