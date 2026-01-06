import glob
import os,sys
from myobsidian import *

#files=['1. Introduction.md']+ ['main.md']

files=['Development-of-Amblyopia.md','Treatments-for-Amblyopia.md',]
texconfig='config/sn-article-template.tex'

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


def md_md(path_from):
    S=parse_include_links(path_from)
    with open(f"docs/_{path_from}",'w') as fid:
        fid.write(S)


def task_md_md():

    for fname in files:
        yield {
            'name': f'Convert {fname} to Markdown with Obsidian links',
            'file_dep': [fname,"myobsidian.py"],
            'targets': [f'docs/_{fname}'],
            'actions': [(md_md, [fname])],
            'verbosity': 2,
            'clean': True,
        }


pandoc= 'pandoc --standalone --csl config/neuron.csl --number-sections -M figPrefix="Figure" --filter pandoc-crossref  --resource-path="resources/" --citeproc'

def task_tex():
    import os
    for fname in files:
        rest,ext=os.path.splitext(fname)

        yield {
            'name': f'Convert {fname} to LaTeX',
            'file_dep': [f'docs/_{fname}',texconfig],
            'targets': [f'docs/{rest}.tex'],
            'actions': [
                f'{pandoc}  docs/_{fname} -o docs/{rest}.tex',
            ],
            'verbosity': 2,
            'clean': True,
        }



def task_pdf():
    import os
    for fname in files:
        rest,ext=os.path.splitext(fname)

        yield {
            'name': f'Convert {fname} to PDF',
            'file_dep': [f'docs/_{fname}',texconfig],
            'targets': [f'docs/{rest}.pdf'],
            'actions': [
                f'{pandoc}  docs/_{fname} -o docs/{rest}.pdf',
            ],
            'verbosity': 2,
            'clean': True,
        }


def task_docx():
    import os
    for fname in files:
        rest,ext=os.path.splitext(fname)

        yield {
            'name': f'Convert {fname} to DOCX',
            'file_dep': [f'docs/_{fname}',texconfig],
            'targets': [f'docs/{rest}.docx'],
            'actions': [
                f'{pandoc}  docs/_{fname} -o docs/{rest}.docx',
            ],
            'verbosity': 2,
            'clean': True,
        }
