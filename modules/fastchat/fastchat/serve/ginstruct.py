# Notice: This file is copied from cli.py and is modified by Jiangyan Feng.
# Copyright 2024 Jiangyan Feng.
# 
#    Licensed under the Creative Commons Attribution-NonCommercial International License, Version 4.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://creativecommons.org/licenses/by-nc/4.0/


"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0

Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
"""
import argparse
import os
import re
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from fastchat.model.model_adapter import add_model_args
from fastchat.modules.gptq import GptqConfig
from fastchat.serve.inference import ChatIO, gen_instruction


class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        # pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            # now = len(output_text) - 1
            # if now > pre:
            #     print(" ".join(output_text[pre:now]), end=" ", flush=True)
            #     pre = now
        # print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)
    
    def complete_output(self, outputs):
        print(outputs.strip())
        return outputs.strip()


class RichChatIO(ChatIO):
    def __init__(self):
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(
            words=["!!exit", "!!reset"], pattern=re.compile("$")
        )
        self._console = Console()

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role}:")

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        #  above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                if not outputs:
                    continue
                text = outputs["text"]
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return text


class ProgrammaticChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        print(f"[!OP:{role}]: ", end="", flush=True)
        contents = ""
        # `end_sequence` is a randomly-generated, 16-digit number
        #  that signals the end of a message. It is unlikely to occur in
        #  message content.
        end_sequence = "9745805894023423"
        while True:
            if len(contents) >= 16:
                last_chars = contents[-16:]
                if last_chars == end_sequence:
                    break
            try:
                char = sys.stdin.read(1)
                contents = contents + char
            except EOFError:
                continue
        return contents[:-16]

    def prompt_for_output(self, role: str):
        print(f"[!OP:{role}]: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)


def main(args, samples=None):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    try:
        results = gen_instruction(
                args.model_path,
                args.device,
                args.num_gpus,
                args.max_gpu_memory,
                args.load_8bit,
                args.cpu_offloading,
                args.conv_template,
                args.temperature,
                args.repetition_penalty,
                args.max_new_tokens,
                SimpleChatIO(),
                GptqConfig(
                    ckpt=args.gptq_ckpt or args.model_path,
                    wbits=args.gptq_wbits,
                    groupsize=args.gptq_groupsize,
                    act_order=args.gptq_act_order,
                ),
                args.revision,
                args.debug,
                args.task_des,
                args.seed_path,
                args.out_path,
                args.start_idx,
                args.stride,
                args.task,
                sample_path=args.sample_path,
                repeat=args.repeat,
                samples=samples,
                dump=args.dump
            )
        return results
    except KeyboardInterrupt:
        print("Error encountered, exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="dump task results",
    )
    parser.add_argument(
        "--task_des",
        type=str,
        default="./prompts/instruct4.txt",
        help="Task description",
    )
    parser.add_argument(
        "--seed_path",
        type=str,
        default="./seed",
        help="Seed examples for in-context learning",
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default="./annotations",
        help="Examples to generate instructions",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./instructions",
        help="path to save generated instructions",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="the start index of sample list",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2000,
        help="the stride of sample list, end index = start index + stride",
    )
    parser.add_argument(
        "--task",
        type=str,
        default='gen_instruct'
    )
    args = parser.parse_args()
    main(args)
