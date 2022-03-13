# src : https://discuss.streamlit.io/t/cannot-print-the-terminal-output-in-streamlit/6602
from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import sys


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


with st_stdout("code"):
    print("Prints as st.code()")

with st_stdout("info"):
    print("Prints as st.info()")

with st_stdout("markdown"):
    print("Prints as st.markdown()")

with st_stdout("success"), st_stderr("error"):
    print("You can print regular success messages")
    print("And you can redirect errors as well at the same time", file=sys.stderr)