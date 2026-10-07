import os
import re
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel


app = FastAPI()

# Allow any origin: the evaluator calls from the browser, from an origin we
# can't predict. allow_credentials must be False when using "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------
def execute_python_code(code: str) -> dict:
    """Run code with exec(); return exact stdout, or the traceback on error."""
    old_stdout = sys.stdout
    buffer = StringIO()
    sys.stdout = buffer

    try:
        exec(code, {"__name__": "__main__"})
        return {"success": True, "output": buffer.getvalue()}

    except BaseException:  # includes SystemExit from exit()/sys.exit()
        etype, evalue, tb = sys.exc_info()
        # Drop this file's own exec() frame so the traceback only shows the
        # user's code (File "<string>", line N), like the example response.
        tb = tb.tb_next if tb is not None else None
        output = "".join(traceback.format_exception(etype, evalue, tb))
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


def lines_from_traceback(tb_text: str) -> List[int]:
    """Deepest line in the user's code, read straight from the traceback."""
    nums = re.findall(r'File "<string>", line (\d+)', tb_text)
    return [int(nums[-1])] if nums else []


# ---------------------------------------------------------------------------
# AI agent (only called on error)
# ---------------------------------------------------------------------------
def analyze_error_with_ai(code: str, tb_text: str) -> List[int]:
    token = os.environ.get("AIPIPE_TOKEN")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set.")

    client = OpenAI(api_key=token, base_url="https://aipipe.org/openrouter/v1")

    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(code.splitlines(), 1))

    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) in the CODE where the error occurred.
Lines in the traceback marked File "<string>" refer to the CODE below.
Use the numbering shown on the left of each code line.

Return ONLY JSON: {{"error_lines": [<int>, ...]}}

CODE:
{numbered}

TRACEBACK:
{tb_text}
"""

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content or ""
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return ErrorAnalysis.model_validate_json(content).error_lines


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {"error": [], "result": execution["output"]}

    traceback_lines = lines_from_traceback(execution["output"])

    try:
        ai_lines = analyze_error_with_ai(request.code, execution["output"])
    except Exception as e:  # never let an AI failure turn into a 500
        print(f"AI analysis failed: {e}", file=sys.stderr)
        ai_lines = []

    # The traceback line is ground truth; use it if the AI disagrees or fails.
    if traceback_lines and traceback_lines[0] not in ai_lines:
        error_lines = traceback_lines
    else:
        error_lines = ai_lines or traceback_lines

    return {"error": error_lines, "result": execution["output"]}


@app.get("/")
def root():
    return {"message": "Code Interpreter API is running"}
