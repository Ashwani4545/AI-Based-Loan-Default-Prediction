import re

path = r'webapp/templates/index.html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

def replacer(match):
    div_start = match.group(1)
    label_inner = match.group(2)
    input_tag = match.group(3)
    
    # Extract placeholder if it exists
    placeholder_match = re.search(r'placeholder=\"([^\"]+)\"', input_tag)
    if placeholder_match:
        placeholder = placeholder_match.group(1)
        # if the label doesn't already have the bracket, add it
        if '[' not in label_inner:
            # Insert it before the <span class="req">*</span> or at the end
            if '<span' in label_inner:
                label_inner = label_inner.replace('<span', f' [{placeholder}] <span')
            else:
                label_inner += f' [{placeholder}]'
                
    return f'{div_start}<label>{label_inner}</label>\n{input_tag}'

pattern = r'(<div class=\"field\">\s*)<label>(.*?)</label>\s*(<input[^>]+>)'
new_content = re.sub(pattern, replacer, content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Updated index.html successfully')
