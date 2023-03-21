import json
import mechanicalsoup
import tqdm
from lxml import html

# Connect to the form's URL
browser = mechanicalsoup.StatefulBrowser()

# Select the form and fill it out

file_list = ['dev.jsonl', 'test.jsonl', 'train.jsonl', ]
dir_list = ['R1', 'R2', 'R3']
file_list = ['anli_v1.0/{}/{}'.format(d, f) for d in dir_list for f in file_list]

for file in file_list:
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    with open(file.replace('.jsonl', '.out.jsonl'), 'w', encoding='utf-8') as f:
        cache = dict()
        new_data = []
        idx = 0
        for item in data:
            item_data = json.loads(item)
            if cache and item_data["context"] == cache["context"]:
                item_data['premise_words'] = cache['premise_words']
                f.write(json.dumps(item_data) + '\n')
                continue
            try:
                browser.open("http://138.197.118.157:8000/segbot/")
                browser.select_form('form[method="post"]')
                browser["inputtext"] = item_data['context']
                # Submit the form
                response_html = browser.submit_selected()

                # Parse the HTML document
                doc = html.fromstring(response_html.text)
                elements = doc.xpath('//form/div/div/div/font')
                count = 0
                words = []
                for element in elements:
                    words.append(element.text.strip().split())
                item_data['premise_words'] = words
                cache = item_data
                idx += 1
                f.write(json.dumps(item_data) + '\n')
            except:
                raise Exception("fail to parse ({}): \"{}\"".format(idx, item_data['context']))
