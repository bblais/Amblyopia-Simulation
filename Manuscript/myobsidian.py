file_extensions=['.pdf','.png','.html','.gif','.svg']

def full_path(fname,tag):
    from pathlib import Path
    import os
    
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
    import os
    
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
    
