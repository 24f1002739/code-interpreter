import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel


app = FastAPI()


# Enable CORS for the evaluator/browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://exam.sanand.workers.dev",
        "https://code-interpreter-2cdu.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


# Request model
class CodeRequest(BaseModel):
    code: str


# Structured AI response
class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return the exact stdout output
    or the exact traceback if execution fails.
    """

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer

    try:
        exec(code, {})

        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()

        # Preserve normal execution output
        output = stdout + stderr

        return {
            "success": True,
            "output": output
        }

    except Exception:
        # Return the complete Python traceback
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def analyze_error_with_ai(code: str, error_output: str) -> List[int]:
    """
    Use AI Pipe + Gemini to identify the exact error line numbers.
    """

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set.")

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openrouter/v1"
    )

    prompt = f"""
Analyze the following Python code and its traceback.

Identify the exact line number or line numbers in the submitted
Python code where the error occurred.

Return ONLY JSON in this exact format:

{{"error_lines": [3]}}

CODE:
{code}

TRACEBACK:
{error_output}
"""

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("AI returned an empty response.")

    result = ErrorAnalysis.model_validate_json(content)

    return result.error_lines


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    """
    Execute the submitted Python code.
    If execution succeeds, return the exact output.
    If execution fails, use AI to identify the error line(s).
    """

    execution = execute_python_code(request.code)

    # Successful code: do NOT call AI
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Error occurred: call AI only now
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }


@app.get("/")
def root():
    return {
        "message": "Code Interpreter API is running"
    }
