import sys
import os
from functools import reduce
import re
import json


INPUTS = [
    "export_dokidokiliterature.json",
    "export_ediea2ndera.json",
    "export_hellotavern.json",
    "export_badlands.json",
]

OUTPUT = "messages.txt"

URL_PATTERN = re.compile(
    "((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)", flags=re.I
)

BOT_TRIGGER_PATTERN = re.compile("^(y|m)!(trigger)?", flags=re.I)


def check_file(path):
    return os.path.isfile(os.path.abspath(path))


def clean_message(accum, message):
    if not message["author"]["isBot"]:
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
