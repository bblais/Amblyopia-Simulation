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


def handle_unicode_error(line,fname):
    if any([ord(x)>127 for x in line]):
        S=fname+" : "
        for x in line:
            if ord(x)<=127:
                S+=x
            else:
                S+=" [%d] " % ord(x)
        raise ValueError("Unicode in %s" % S)


def parse_include_links_broken(input_file, image_dir="resources"):
    import re
    from pathlib import Path
    import sys

    input_path = Path(input_file)
    image_base = Path(image_dir).resolve()

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match first line: Obsidian-style image
        match_image = re.match(r'!\[\[(.*?)\]\]', line)
        if match_image:
            image_name = match_image.group(1)

            # Try to find {#fig:...} on this or next line
            fig_id = None
            caption = None

            # Look ahead safely
            if (i + 1) < len(lines):
                next_line = lines[i+1].strip()
                id_match = re.match(r'\{#(fig:[\w\-]+)\}', next_line)
                if id_match:
                    fig_id = id_match.group(1)
                    i += 1  # Consume the fig ID line

            # Look for caption on next line (after ID or image line)
            if (i + 1) < len(lines):
                caption_line = lines[i+1].strip()
                if caption_line.startswith(">"):
                    caption = caption_line[1:].strip()
                    i += 1  # Consume the caption line

            # If we got both ID and caption, format image line
            if fig_id and caption:
                full_image_path = image_base / image_name
                output_line = f"![{caption}]({full_image_path}){{#{fig_id}}}"
                output_lines.append(output_line)
                i += 1  # Move past the image line
                continue  # Skip to next iteration

        # If not matched, keep original line
        output_lines.append(lines[i].rstrip())
        i += 1

    S="\n".join(output_lines) + "\n"

    return S



def parse_include_links(input_file, image_dir="resources"):
    import re
    from pathlib import Path
    import sys

    
    input_path = Path(input_file)
    image_base = Path(image_dir).resolve()

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match Obsidian-style image with ID
        match = re.match(r'!\[\[(.*?)\]\]\s*\{#(fig:[\w\-]+)\}', line)
        if match and (i + 1) < len(lines) and lines[i+1].strip().startswith(">"):
            image_name, fig_id = match.groups()
            caption = lines[i + 1].strip()[1:].strip()  # Remove '>'

            full_image_path = image_base / image_name
            output_line = f"![{caption}]({full_image_path}){{#{fig_id}}}"
            output_lines.append(output_line)
            i += 2  # Skip next line (caption)
        else:
            output_lines.append(lines[i].rstrip())
            i += 1

    S="\n".join(output_lines) + "\n"

    return S


def parse_include_linksold(fname):
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

    embed="![["

    for line in lines:
        handle_unicode_error(line,fname)

        if look_for_caption:
            if line.startswith('>'):
                caption.append(line[1:])
                continue
            elif line.strip():  # non-empty line
                caption.append(line)
                continue
            else:  # blank line
                look_for_caption=False
                if caption:
                    captions[tag]="\n".join(caption)
                    caption=[]
        
        if not embed in line:
            new_lines.append(line)
            continue
        
        part=line

        assert line.count("]]")==1  # only one include per line -- is there a reason to do another?


        tag=line.split("[[")[1].split(']]')[0]
        path=full_path(fname,tag)
        base,ext=os.path.splitext(path)

        if ext in file_extensions:  # a figure
            look_for_caption=True
            caption=[]   

            line=line.replace("![[%s]]" % tag,
                              f'![{tag}]({path})')  # change ![[filename.png]] to ![filename.png](/full/path/to/filename.png)
        elif ext=='.md':  # a markdown file
            line=line.replace("![[%s]]" % tag,parse_include_links(path)) # change ![[mdfilename]] to "full contents of mdfilename")           
        else:
            raise("You can't get there from here.")
            
        # example
        #![Simple model of a neuron with 4 inputs ($x_1, x_2, x_3,$ and $x_4$), connecting to the cell via 4 synaptic weights ($w_1, w_2, w_3,$ and $w_4$), yielding an output ($y$).](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Simple Neuron.pdf){#fig:simple-neuron-pdf}
                
        new_lines.append(line)    
        
    # S='\n'.join(new_lines)
    # for tag in captions:
    #     text=captions[tag]
    #     assert text.count("{#")<=1  # no more than one reference

        
    #     if "{#" in text:
    #         idx0=text.index("{#")
    #         idx1=text[idx0:].index("}")+idx0+1
    #         full_ref=text[idx0:idx1]
    #         ref=full_ref
    #         text=text.replace(full_ref,'')
    #     else:
    #         ref='{#fig:%s}' % tag.replace(" ","_") 
            

    #     S=S.replace(f'{{#figref:{tag}}}',ref)
    #     S=S.replace(f"[{tag}]",f"[{text}]")
        
            
        
    return S
    