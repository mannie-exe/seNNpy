from sys import exit as sysexit
from os.path import abspath
import re
import argparse
import asyncio
import discord
import modules.generate as generate


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="seNNpy is the Discord bot of your hot, wet dreams"
    )
    parser.add_argument(
        "--train",
        type=int,
        action="store",
        help="if >1, training for that many epochs before continuing",
    )
    return parser.parse_args()


async def init_bot(bot_token):
    TRIGGER_PATTERN = re.compile("^hey senna((-| )?chan)?", flags=re.I)

    client = discord.Client()

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        if re.match(TRIGGER_PATTERN, message.content):
            response = generate.text("hey {}, ".format(message.author), count=512)
            response = response.split("\n")
            # await message.channel.send(response[0])

    print("Connecting seNNpy with token: {}".format(bot_token))
    try:
        await client.login(bot_token)
        await client.connect()
    except:
        print("Wellp, that's not good. Goodbye! x.x")
    print("seNNpy is online 😎")


if __name__ == "__main__":
    args = parse_arguments()

    TOKEN = abspath("./BOT_TOKEN")
    with open(TOKEN) as token_file:
        TOKEN = token_file.read().strip()
        if not TOKEN:
            sysexit(
                "You fucked up. Try making a Discord bot and placing the token in './BOT_TOKEN', you simple bitch."
            )

    if args.train:
        import modules.training as training

        training.run(args.train)

    # this procs TensorFlow, making it quicker next few times
    generate.text("dummy")

    asyncio.run(init_bot(TOKEN))
