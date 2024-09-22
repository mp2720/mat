import os
import re
import operator

from functools import reduce
from xml.etree.ElementTree import Element, SubElement

from markdown import markdown
from markdown.blockprocessors import BlockProcessor
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor
from markdown.util import AtomicString


class InlineLatexProcessor(InlineProcessor):
    def __init__(self, md):
        super().__init__(r"\$(.+?)\$", md)

    def handleMatch(self, m: re.Match[str], data) -> tuple[Element, int, int] | tuple[None, None, None]:
        # Make sure there's only one dollar sign.
        if m.group(0).startswith('$$'):
            return None, None, None

        el = Element("span")
        el.attrib["class"] = "latex-inline"
        el.text = AtomicString(r"\(" + m.group(1) + r"\)")
        return el, m.start(0), m.end(0)


class CustomCodeBlockProcessor(BlockProcessor):
    def test(self, parent, block):
        return block.lstrip().startswith("```")

    def run(self, parent: Element, blocks: list[str]):
        # Cut the opening '```' from the first block.
        first_block = blocks[0].lstrip()[3:]

        cut_blocks = [first_block] + blocks[1:]

        for block_num, block in enumerate(cut_blocks):
            if block.rstrip().endswith("```"):
                el = SubElement(parent, "pre")
                el.set("class", "code-block")
                el.text = AtomicString(reduce(operator.add, cut_blocks[0:block_num + 1]).rstrip()[:-3])

                for _ in range(0, block_num + 1):
                    blocks.pop(0)
                return True

        return False


class Ext(Extension):
    def extendMarkdown(self, md):
        md.inlinePatterns.register(InlineLatexProcessor(md), "inline-latex", 189)
        md.parser.blockprocessors.register(CustomCodeBlockProcessor(md.parser), "latex-block", 175)


def render_page(text: str, page_name: str) -> str:
    text = markdown(text, extensions=['tables', Ext()])
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width">
        <title>""" + page_name + """</title>
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6">
        </script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            .main {
                margin: auto;
                max-width: 1000px;
            }
        </style>
    </head>
    <body>
        <div class="contents">
        </div>
        <div class="main">
    """ + text + """ 
        </div>
    </body>
    """


CONTENTS = {
    "log" : (
        "Логика",
        {
            "pl.md": "Логика высказываний",
        }
    ),
    "field.md": "Поле",
    "set.md": "Теория множеств",
}

HTML_DIR_PATH = "html/"


def render_repo():
    if not os.path.exists(HTML_DIR_PATH):
        os.mkdir(HTML_DIR_PATH)

    def render_entry(parent_path: str, entries: dict):
        for entry_filename, entry in entries.items():
            if type(entry) == tuple:
                entry_name = entry[0]
                children_entries = entry[1]
                dir_path = parent_path + entry_filename + '/'
                if not os.path.exists(HTML_DIR_PATH + dir_path):
                    os.mkdir(HTML_DIR_PATH + dir_path)

                render_entry(dir_path, children_entries)
            else:
                entry_name = entry
                with open(parent_path + entry_filename, "r") as f:
                    text = f.read()

                text = render_page(text, entry_name)
                print(f"{entry_name} {parent_path+entry_filename} rendered")

                with open(HTML_DIR_PATH + parent_path + entry_filename + ".html", "w") as f:
                    f.write(text)

    render_entry("", CONTENTS)

render_repo()

