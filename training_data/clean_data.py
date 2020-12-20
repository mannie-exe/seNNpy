import sys
import os
from functools import reduce
import re
import json


raw_messages_path = os.path.abspath('export.json')
raw_messages_exist = os.path.isfile(raw_messages_path)
if not raw_messages_exist:
    sys.exit("Use DiscordChatExporter and export a JSON format message list as 'export.json', and paste the file here")

def clean_messages(accum, message):
    URL_PATTERN = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', flags=re.I)
    if not message['author']['isBot']:
        if re.match(URL_PATTERN, message['content']):
            message['content'] = re.sub(URL_PATTERN, '', message['content'])
        if message['content'].strip():
            accum.append(message['content'])
    return accum


with open(raw_messages_path, encoding='utf-8') as raw_messages:
    raw_messages = json.loads(raw_messages.read())['messages']
    clean_messages = reduce(clean_messages, raw_messages, [])

    new_file_path = os.path.abspath('messages.txt')
    with open(new_file_path, 'w', encoding='utf-8') as new_messages_file:
        new_messages_file.write('\n'.join(clean_messages))

    print('%d -> %d' % (len(raw_messages), len(clean_messages)))

