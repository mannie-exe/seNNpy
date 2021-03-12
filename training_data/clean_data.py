import sys
import os
from functools import reduce
import re
import json


INPUTS = [
    "./training_data/export_dokidokiliterature.json",
    "./training_data/export_ediea2ndera.json",
    "./training_data/export_hellotavern.json",
    "./training_data/export_badlands.json",
]

OUTPUT = "./training_data/messages.txt"

URL_PATTERN = re.compile(
    "((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)", flags=re.I
)

BOT_TRIGGER_PATTERN = re.compile(
    "(^(y|m|s)!(trigger)?|^hey senna((-| )?chan)?)", flags=re.I
)


def check_file(path):
    return os.path.isfile(os.path.abspath(path))


def clean_message(accum, message):
    bot = message["author"]["isBot"]
    old_markov = int(message["author"]["id"]) == 569277281046888488
    if bot and not old_markov:
        return accum

    message = message["content"]
    message = re.sub(URL_PATTERN, "", message).strip()
    message = "" if re.match(BOT_TRIGGER_PATTERN, message) else message
    if message:
        accum.append(message)
    return accum


def write_clean_messages(messages):
    path = os.path.abspath(OUTPUT)

    if os.path.isfile(path):
        os.remove(path)

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(messages))


def main():
    for file in INPUTS:
        if not check_file(file):
            sys.exit(
                "Missing raw, message file! Use DiscordChatExporter.\nExpected file: {}".format(
                    file
                )
            )

    all_raw_messages = []
    all_clean_messages = []
    for input in INPUTS:
        path = os.path.abspath(input)
        with open(path, "r", encoding="utf-8") as raw_json:
            raw_messages = json.loads(raw_json.read())["messages"]
            clean_messages = reduce(clean_message, raw_messages, [])

            print(
                "Cleaned {}: {} <- {}".format(
                    input, len(clean_messages), len(raw_messages)
                )
            )

            all_raw_messages += raw_messages
            all_clean_messages += clean_messages

    print(
        "Cleaned everything: {} <- {}".format(
            len(all_clean_messages), len(all_raw_messages)
        )
    )

    write_clean_messages(all_clean_messages)


main()
