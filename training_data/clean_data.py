import sys
import os
from functools import reduce
import re
import json


URL_PATTERN = re.compile(
    "((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)", flags=re.I
)

raw_messages_path = os.path.abspath("export.json")
raw_messages_exist = os.path.isfile(raw_messages_path)
if not raw_messages_exist:
    sys.exit(
        "Use DiscordChatExporter to export a JSON format message list as 'export.json', then paste the file here"
    )


def clean_messages(accum, message):
    if not message["author"]["isBot"]:
        message["content"] = re.sub(URL_PATTERN, "", message["content"]).strip()
        if message["content"]:
            accum.append(message["content"])
    return accum


with open(raw_messages_path, encoding="utf-8") as raw_messages:
    raw_messages = json.loads(raw_messages.read())["messages"]
    clean_messages = reduce(clean_messages, raw_messages, [])

    new_file_path = os.path.abspath("messages.txt")
    if os.path.isfile(new_file_path):
        os.remove(new_file_path)
    with open(new_file_path, "w", encoding="utf-8") as new_messages_file:
        new_messages_file.write("\n".join(clean_messages))

    print(
        "Cleaned '{}' messages down to '{}' messages".format(
            len(raw_messages), len(clean_messages)
        )
    )
