from doit.tools import run_once
import glob
import os
    
files=glob.glob('[0-9]*.md') + ['main.md']

target_name='article'

import sys
file_extensions=['.pdf','.png','.html','.gif']

def full_path(fname,tag):
    from pathlib import Path
    
    base,rest=os.path.split(fname)
    if not base:
        base="."

    tag_base,tag_ext=os.path.splitext(tag)


    if tag_ext.lower() not in file_extensions:  # markdown file
        tag=tag+".md"
        
    parts=Path(tag).parts
    new_fname=None
    if len(parts)==1:  # not a relative path
        for root, dirs, files in os.walk(base):
            if tag in files:
                new_fname=os.path.join(root, tag)
                break
    else:
        new_fname=os.path.join(base)

    assert not new_fname is None

    return os.path.abspath(new_fname)

def parse_include_links(fname):
    
    if fname.startswith("!"):  # not a file
        return fname
    
    with open(fname) as fid:
        S=fid.read()
    
    lines=S.split('\n')
    
    includes=[]
    new_lines=[]
    
    D={}
    captions={}
    caption=[]
    look_for_caption=False
    
    for line in lines:
        if look_for_caption:
            if line.startswith('>'):
                caption.append(line[1:])
                continue
            elif line.startswith("--"):
                continue
            elif line.strip():
                caption.append(line)
                continue
            else:  # blank line
                look_for_caption=False
                if caption:
                    captions[tag]="\n".join(caption)
                    caption=[]
        
        
        if not "![[" in line:
            new_lines.append(line)
            continue

        part=line
            
        while "![[" in part:
            idx1=part.index("![[")+3
            idx2=part[idx1:].index(']]')+idx1
            tag=part[idx1:idx2]
            includes.append(tag)
            part=part[idx2+2:]
            
            if not tag in D:
                path=full_path(fname,tag)

                base,ext=os.path.splitext(path)
                if ext in file_extensions:
                    D[tag]=f'![{tag}]({path})'
                else:
                    D[tag]=path

            line=line.replace("![[%s]]" % tag,parse_include_links(D[tag]))
                
        base,ext=os.path.splitext(path)
        if ext in file_extensions:   
            look_for_caption=True
            caption=[]
            
        
                
                
        new_lines.append(line)    
        
    S='\n'.join(new_lines)
    for tag in captions:
        S=S.replace(f"[{tag}]",f"[{captions[tag]}]")
            
        
        
    
    
    return S
    


def md_md(path_from,path_to):
    S=parse_include_links(path_from)
    with open(path_to,'w') as fid:
        fid.write(S)


def task_prelim():
    return {
            'targets': ['build'], # files produced
            'actions': ['mkdir build'],
            'uptodate': [run_once],
            'clean':True,
    }    


def task_md_md():
    return {
        'file_dep': files ,
        'targets': ['build/_main.md'],
        'actions': md_md('main.md','build/_main.md'),
        'verbosity':2,
        'clean':True,
    }


def task_md_tex():
    import os

    return {
        'file_dep': ['build/_main.md'],
        'targets': ['build/%s.tex' % target_name],
        'actions': [f'pandoc  --number-sections --template=config/paper_template.tex -M fignos-warning-level=1 --filter pandoc-fignos  --filter pandoc-secnos --resource-path="resources/" --citeproc  build/_main.md -o build/{target_name}.tex'],
        'clean':True,
    }

def task_md_docx():
    import os

    return {
        'file_dep': ['build/_main.md'],
        'targets': ['build/%s.docx' % target_name],
        'actions': [f'pandoc  --number-sections -M fignos-warning-level=1 --filter pandoc-fignos  --filter pandoc-secnos --resource-path="resources/" --citeproc  build/_main.md -o build/{target_name}.docx'],
        'clean':True,
    }

def task_md_pdf():
    import os

    return {
        'file_dep': ['build/_main.md'],
        'targets': ['build/%s.pdf' % target_name],
        'actions': [f'pandoc  --number-sections --template=config/paper_template.tex -M fignos-warning-level=1 --filter pandoc-fignos --filter pandoc-secnos --resource-path="resources/" --citeproc  build/_main.md -o build/{target_name}.pdf'],
        'clean':True,
    }


from doit.task import clean_targets    
# def task_tex_pdf():
#     return {
#         'file_dep': ['build/%s.tex' % target_name,'stddefs.tex'],
#         'targets': ['build/%s.pdf' % target_name],
#         'actions': ['pdflatex -halt-on-error -output-directory build "build/%s.tex"' % target_name, 
#                     'pdflatex -halt-on-error -output-directory build "build/%s.tex"'  % target_name, 
#                     'cp build/%s.pdf .' % target_name],
#         'clean':[clean_targets,'rm -f build/*.log build/*.aux build/*.toc build/*.out build/*.bbl build/*.blg'],
#         'verbosity': 2,
#     }




        