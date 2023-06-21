import glob
import os,sys
from myobsidian import *

files=glob.glob('[0-9]*.md') + ['main.md']
target_name='Comparing-Treatments-for-Amblyopia-with-a-Synaptic-Plasticity-Model'


def dir_exists(task):
    directory=task.targets[0]
    return (os.path.exists(directory) and os.path.isdir(directory))


def task_prelim():
    return {
            'targets': ['docs'], # files produced
            'actions': ['mkdir docs'],
            'uptodate': [dir_exists],
            'clean':True,
    }    


def md_md(path_from,path_to):
    S=parse_include_links(path_from)
    with open(path_to,'w') as fid:
        fid.write(S)


def task_md_md():
    return {
        'file_dep': files,
        'targets': ['docs/_main.md'],
        'actions': [(md_md, ['main.md','docs/_main.md'])],
        'verbosity':2,
        'clean':True,
    }


def task_tex():
    import os

    return {
        'file_dep': ['docs/_main.md','config/paper_template.tex'],
        'targets': ['docs/%s.tex' % target_name],
        'actions': [f'pandoc  --number-sections --template=config/paper_template.tex -M xnos-warning-level=1 --filter pandoc-xnos  --resource-path="resources/" --citeproc  docs/_main.md -o docs/{target_name}.tex'],
        'clean':True,
    }

def task_docx():
    import os

    return {
        'file_dep': ['docs/_main.md'],
        'targets': ['docs/%s.docx' % target_name],
        'actions': [f'pandoc  --toc --number-sections -M xnos-warning-level=1 --filter pandoc-xnos  --resource-path="resources/" --citeproc  docs/_main.md -o docs/{target_name}.docx'],
        'clean':True,
    }

def task_html():
    import os

    return {
        'file_dep': ['docs/_main.md'],
        'targets': ['docs/%s.html' % target_name],
        'actions': [f'pandoc  --toc  -s --mathjax  --number-sections -M xnos-warning-level=1 --filter pandoc-xnos   --resource-path="resources/" --citeproc  docs/_main.md -o docs/{target_name}.html'],
        'clean':True,
    }

def task_pdf():
    import os

    return {
        'file_dep': ['docs/_main.md','config/paper_template.tex'],
        'targets': ['docs/%s.pdf' % target_name],
        'actions': [f'pandoc  --number-sections --template=config/paper_template.tex -M xnos-warning-level=1 --filter pandoc-xnos   --resource-path="resources/" --citeproc  docs/_main.md -o docs/{target_name}.pdf'],
        'clean':True,
    }
