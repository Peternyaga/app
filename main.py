# file: exam_pipeline_app.py
import os
import uuid
import json
import re
from typing import Dict, List, Any, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from ocr_utils import ocr_image, ocr_pdf
from fastapi.middleware.cors import CORSMiddleware



# ----------------- Configuration -----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY environment variable")

# Use the model endpoint as appropriate. Prefer Authorization header if supported.
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


# LLM generation config defaults
DEFAULT_GEN_CONFIG = {
    "max_output_tokens": 2000,
    "temperature": 0.2,
    "topP": 1
}

# ----------------- FastAPI app -----------------
app = FastAPI(title="Stateful Exam Marking Pipeline (OCR + LLM formatting + evaluation)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for marking schemes. { scheme_id: {questions: [...], metadata: {...}} }
# NOTE: replace with DB (Redis/Postgres) in production for persistence and scaling.
marking_schemes: Dict[str, Dict[str, Any]] = {}

# ----------------- Pydantic models -----------------
class QuestionSchema(BaseModel):
    number: int
    question: str
    max_marks: Optional[int] = None
    marking_scheme: Optional[str] = None

class AnswerSchema(BaseModel):
    number: int
    question: Optional[str] = None
    student_response: Optional[str] = None

class EvaluationItem(BaseModel):
    number: int
    score: int
    comment: str

# ----------------- Helper: LLM call -----------------
async def call_gemini_system_prompt(prompt_text: str, gen_config: dict = None) -> str:
    """
    Generic call to Gemini. Returns the raw text output from the model.
    Raises HTTPException on non-200 or parsing issues.
    """
    gen_config = gen_config or DEFAULT_GEN_CONFIG

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "systemInstruction": {
            "role": "systemInstruction",
            "parts": [{"text": "You are a helpful formatter and evaluator. Return JSON only unless asked otherwise."}]
        },
        "generation_config": gen_config
    }

    # headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(GEMINI_URL, json=payload)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"LLM request error: {exc}") from exc

    if r.status_code != 200:
        # attempt to provide helpful error message
        raise HTTPException(status_code=502, detail=f"LLM returned {r.status_code}: {r.text}")

    data = r.json()
    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Unexpected LLM response shape: {e} | {data}")

    # Clean fencing if any (robust)
    clean = raw_text.strip()
    # remove Markdown code fences around JSON if present
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean, flags=re.IGNORECASE)
    return clean

# ----------------- LLM format helpers -----------------
async def format_questions_with_llm(ocr_text: str) -> List[Dict[str, Any]]:
    """
    Use LLM to extract questions in JSON:
    [ {'number': <int>, 'question': '<text>', 'max_marks': <int>, 'marking_scheme': '<scheme>'}, ... ]
    """
    prompt = (
        "You are given OCR text from an exam question paper. Extract each question and its "
        "marking information and return valid JSON only with the format:\n\n"
        "[\n  {\"number\": <int>, \"question\": \"<text>\", \"max_marks\": <int>, \"marking_scheme\": \"<scheme>\"},\n  ...\n]\n\n"
        "If max marks or marking_scheme are missing, set max_marks to null and marking_scheme to an empty string.\n\n"
        "Here is the OCR text:\n\n"
        f"{ocr_text}\n\nReturn JSON only."
    )

    clean = await call_gemini_system_prompt(prompt, gen_config={"max_output_tokens": 2000, "temperature": 0.0, "topP": 1})
    try:
        questions = json.loads(clean)
    except json.JSONDecodeError as e:
        # Show snippet with helpful error
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON for questions. Raw output: {clean[:1000]!r}")
    # Basic validation
    validated = []
    for q in questions:
        # Defensive defaults
        validated.append({
            "number": int(q.get("number")),
            "question": q.get("question", "").strip(),
            "max_marks": int(q["max_marks"]) if q.get("max_marks") not in (None, "") else None,
            "marking_scheme": q.get("marking_scheme", "") or ""
        })
    return validated

async def format_answers_with_llm(ocr_text: str) -> List[Dict[str, Any]]:
    """
    Use LLM to extract student answers:
    [ {'number': <int>, 'question': '<maybe omitted>', 'student_response': '<text>' }, ... ]
    """
    prompt = (
        "You are given OCR text from a student's answer sheet. Extract each numbered answer and return valid JSON only "
        "in the format:\n\n"
        "[\n  {\"number\": <int>, \"question\": \"<optional question excerpt>\", \"student_response\": \"<answer text>\"},\n  ...\n]\n\n"
        "If the question text is not present on the answer sheet, set question to an empty string.\n\n"
        f"Here is the OCR text:\n\n{ocr_text}\n\nReturn JSON only."
    )

    clean = await call_gemini_system_prompt(prompt, gen_config={"max_output_tokens": 1500, "temperature": 0.0, "topP": 1})
    try:
        answers = json.loads(clean)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON for answers. Raw output: {clean[:1000]!r}")
    validated = []
    for a in answers:
        validated.append({
            "number": int(a.get("number")),
            "question": a.get("question", "") or "",
            "student_response": a.get("student_response", "") or ""
        })
    return validated

async def evaluate_with_llm(questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Send merged questions & answers to LLM and receive scores/comments.
    Output format expected:
    [ {'number': <int>, 'score': <int>, 'comment': '<text>'}, ... ]
    """
    # Merge questions/answers by number (include full marking_scheme for each)
    merged = []
    q_map = {q["number"]: q for q in questions}
    for a in answers:
        q = q_map.get(a["number"], {})
        merged.append({
            "number": a["number"],
            "question": q.get("question", a.get("question", "")),
            "max_marks": q.get("max_marks") if q.get("max_marks") is not None else 0,
            "marking_scheme": q.get("marking_scheme", ""),
            "student_response": a.get("student_response", "")
        })

    prompt = (
        "You are an exam marker. For each item in the provided list, score the student's response using the marking "
        "scheme / expected answer. Provide a short constructive comment and a numeric score (integer) between 0 and max_marks. "
        "Return JSON only in this exact structure:\n\n"
        "[\n  {\"number\": <int>, \"score\": <int>, \"comment\": \"<text>\"},\n  ...\n]\n\n"
        "Here is the data:\n\n"
        f"{json.dumps(merged, indent=2)}\n\nReturn JSON only."
    )

    clean = await call_gemini_system_prompt(prompt, gen_config={"max_output_tokens": 2000, "temperature": 0.0, "topP": 1})
    try:
        evaluation = json.loads(clean)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON for evaluation. Raw output: {clean[:1000]!r}")
    # Validate and coerce types
    validated = []
    for e in evaluation:
        validated.append({
            "number": int(e["number"]),
            "score": int(e["score"]),
            "comment": e.get("comment", "")
        })
    return validated

# ----------------- API Endpoints -----------------

@app.post("/upload-question-paper/")
async def upload_question_paper(file: UploadFile = File(...), title: str = Form(None)):
    """
    Upload a question paper (image or pdf). OCR -> LLM formats it into questions.
    Returns a scheme_id that can be used to evaluate many answer sheets.
    """
    contents = await file.read()
    try:
        text = ocr_pdf(contents) if file.filename.lower().endswith(".pdf") else ocr_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR error: {e}")

    questions = await format_questions_with_llm(text)
    scheme_id = str(uuid.uuid4())
    marking_schemes[scheme_id] = {
        "title": title or file.filename,
        "questions": questions
    }
    return {"scheme_id": scheme_id, "structured_questions": questions}

@app.get("/list-schemes/")
async def list_schemes():
    """List scheme ids and titles (in-memory)."""
    return [{"scheme_id": sid, "title": v["title"], "questions_count": len(v["questions"])} for sid, v in marking_schemes.items()]

@app.get("/get-scheme/{scheme_id}")
async def get_scheme(scheme_id: str):
    scheme = marking_schemes.get(scheme_id)
    if not scheme:
        raise HTTPException(status_code=404, detail="scheme_id not found")
    return scheme

@app.post("/evaluate-answer-sheet/")
async def evaluate_answer_sheet(
    file: UploadFile = File(...),
    scheme_id: str = Form(...),
    name: str = Form(...),
    student_id: str = Form(...),
    student_class: str = Form(...)
):
    """
    Upload an answer sheet and evaluate it against an existing scheme_id.
    Returns student details and evaluation items per question.
    """
    if scheme_id not in marking_schemes:
        raise HTTPException(status_code=404, detail="scheme_id not found")

    contents = await file.read()
    try:
        text = ocr_pdf(contents) if file.filename.lower().endswith(".pdf") else ocr_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR error: {e}")

    answers = await format_answers_with_llm(text)
    scheme = marking_schemes[scheme_id]["questions"]
    evaluation = await evaluate_with_llm(scheme, answers)

    # Merge evaluation results back into answers for a full report
    ev_map = {e["number"]: e for e in evaluation}
    annotated_answers = []
    for a in answers:
        item = {
            "number": a["number"],
            "question": a.get("question", ""),
            "student_response": a.get("student_response", ""),
            "score": ev_map.get(a["number"], {}).get("score"),
            "comment": ev_map.get(a["number"], {}).get("comment")
        }
        annotated_answers.append(item)

    return {
        "student_details": {"name": name, "student_id": student_id, "class": student_class},
        "scheme_id": scheme_id,
        "evaluation": annotated_answers
    }

@app.post("/process-exam-file/")
async def process_exam_file(
    question_paper: UploadFile = File(...),
    answer_sheet: UploadFile = File(...),
    name: str = Form(...),
    student_id: str = Form(...),
    student_class: str = Form(...)
):
    """
    Convenience endpoint: upload question paper + answer sheet together (no pre-uploaded scheme required).
    This will create a scheme_id (in-memory) and immediately evaluate the provided answer sheet.
    """
    # OCR + format question paper
    q_contents = await question_paper.read()
    q_text = ocr_pdf(q_contents) if question_paper.filename.lower().endswith(".pdf") else ocr_image(q_contents)
    questions = await format_questions_with_llm(q_text)
    scheme_id = str(uuid.uuid4())
    marking_schemes[scheme_id] = {"title": question_paper.filename, "questions": questions}

    # OCR + format answers
    a_contents = await answer_sheet.read()
    a_text = ocr_pdf(a_contents) if answer_sheet.filename.lower().endswith(".pdf") else ocr_image(a_contents)
    answers = await format_answers_with_llm(a_text)

    evaluation = await evaluate_with_llm(questions, answers)
    ev_map = {e["number"]: e for e in evaluation}
    annotated_answers = []
    for a in answers:
        annotated_answers.append({
            "number": a["number"],
            "question": a.get("question", ""),
            "student_response": a.get("student_response", ""),
            "score": ev_map.get(a["number"], {}).get("score"),
            "comment": ev_map.get(a["number"], {}).get("comment")
        })

    return {
        "scheme_id": scheme_id,
        "student_details": {"name": name, "student_id": student_id, "class": student_class},
        "structured_questions": questions,
        "evaluation": annotated_answers
    }

@app.delete("/delete-scheme/{scheme_id}")
async def delete_scheme(scheme_id: str):
    if scheme_id in marking_schemes:
        del marking_schemes[scheme_id]
        return {"deleted": scheme_id}
    raise HTTPException(status_code=404, detail="scheme_id not found")
