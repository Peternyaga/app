from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from pipeline import process_exam, parse_question_paper, parse_student_answers, parse_answer_only

from ocr_utils import ocr_image, ocr_pdf
import os
import httpx
import json




app = FastAPI(title="Exam Marking API with OCR")

# Load your Gemini API key from env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini endpoint (model = gemini-2.0-flash)
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from app.ocr_utils import ocr_image, ocr_pdf


# app = FastAPI()
questions = []
answers = []

@app.post("/process_question_paper/")
async def upload_file(question_paper: UploadFile = File(...)):
    try:
        contents = await question_paper.read()
        
        if question_paper.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            text = ocr_image(contents)
        elif question_paper.filename.lower().endswith(".pdf"):
            text = ocr_pdf(contents)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # questions = parse_question_paper(text)

        return {"filename": question_paper.filename, "extracted_text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/process_answer_sheet/")
async def upload_file(answer_sheet: UploadFile = File(...)):
    try:
        contents = await answer_sheet.read()
        
        if answer_sheet.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            text = ocr_image(contents)
        elif answer_sheet.filename.lower().endswith(".pdf"):
            text = ocr_pdf(contents)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        answers.append(parse_question_paper(text))

        return {"filename": answer_sheet.filename, "extracted_text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_exam_file")
async def process_exam_file(
    question_paper: UploadFile,
    answer_sheet: UploadFile,
    name: str = Form(...),
    student_id: str = Form(...),
    student_class: str = Form(...)
):
    # OCR question paper
    if question_paper.filename.endswith(".pdf"):
        q_text = ocr_pdf(await question_paper.read())
    else:
        q_text = ocr_image(await question_paper.read())

    # OCR answer sheet
    if answer_sheet.filename.endswith(".pdf"):
        a_text = ocr_pdf(await answer_sheet.read())
    else:
        a_text = ocr_image(await answer_sheet.read())

    # Student details
    student_details = {
        "name": name,
        "student_id": student_id,
        "class": student_class
    }

    # Run pipeline
    report = process_exam(q_text, a_text, student_details)
    return report

# Request model
class ExamRequest(BaseModel):
    question_paper_text: str
    student_answer_text: str
    student_details: dict

@app.post("/process_exam")
def process_exam_endpoint(request: ExamRequest):
    """
    Endpoint that accepts question paper OCR text,
    student answers OCR text, and student details.
    Returns scored report.
    """
    report = process_exam(
        request.question_paper_text,
        request.student_answer_text,
        request.student_details
    )
    return report


@app.post("/format-answers/")
async def format_answers(ocr_text: str):
    """
    Take raw OCR text from an answer sheet and send it to Gemini
    to return structured JSON of {number, question, student_response}.
    """

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"You are given OCR text from a student's exam answer sheet. Extract each question and its corresponding student response, and return them in JSON format only:\n\n[\n  {{'number': <question_number>, 'question': '<question>', 'student_response': '<answer>'}}\n]\n\nHere is the OCR text:\n\n{ocr_text}"
                    }
                ]
            }
        ],
        "systemInstruction": {
            "role": "systemInstruction",
            "parts": [
                {
                    "text": "You are a data formatter. Return valid JSON only, no explanations."
                }
            ]
        },
        "generation_config": {
            "max_output_tokens": 1500,
            "temperature": 0.2,
            "topP": 1
        }
    }

    async with httpx.AsyncClient() as client:
        
        response = await client.post(GEMINI_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    # Extract model output
    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

    # Clean markdown fences if any
    clean_text = raw_text.strip("```json").strip("```").strip()

    return {"structured_answers": json.loads(clean_text)}


# @app.post("/format-answers/")
# async def format_answers(ocr_text: str):
    """
    Take raw OCR text from an answer sheet and send it to Gemini
    to return structured JSON of {number, question, student_response}.
    """
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"You are given OCR text from a student's exam answer sheet. "
                                f"Extract each question and its corresponding student response, and return them in JSON format only:\n\n"
                                f"[{{'number': <question_number>, 'question': '<question>', 'student_response': '<answer>'}}]\n\n"
                                f"Here is the OCR text:\n\n{ocr_text}"
                    }
                ]
            }
        ],
        "systemInstruction": {
            "role": "systemInstruction",
            "parts": [
                {"text": "You are a data formatter. Return valid JSON only, no explanations."}
            ]
        },
        "generation_config": {
            "max_output_tokens": 1500,
            "temperature": 0.2,
            "topP": 1
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    clean_text = raw_text.strip("```json").strip("```").strip()

    return {"structured_answers": json.loads(clean_text)}


@app.post("/format-questions/")
async def format_questions(ocr_text: str):

    """
    Take raw OCR text from a question paper and send it to Gemini
    to return structured JSON of {number, question, max_marks, marking_scheme}.
    """
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"You are given OCR text from an exam question paper with marking scheme. "
                                f"Extract each question and its marking scheme, and return them in JSON format only:\n\n"
                                f"[{{'number': <question_number>, 'question': '<question>', 'max_marks': <marks>, 'marking_scheme': '<scheme>'}}]\n\n"
                                f"Here is the OCR text:\n\n{ocr_text}"
                    }
                ]
            }
        ],
        "systemInstruction": {
            "role": "systemInstruction",
            "parts": [
                {"text": "You are a data formatter. Return valid JSON only, no explanations."}
            ]
        },
        "generation_config": {
            "max_output_tokens": 2000,
            "temperature": 0.2,
            "topP": 1
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    clean_text = raw_text.strip("```json").strip("```").strip()

    questions = json.loads(clean_text)
    return {"structured_questions": questions}


from pydantic import BaseModel
from typing import List, Optional

class Question(BaseModel):
    number: int
    question: str
    max_marks: int
    marking_scheme: str

class Answer(BaseModel):
    number: int
    question: str
    student_response: Optional[str] = None

class EvaluationRequest(BaseModel):
    questions: List[Question]
    answers: List[Answer]


@app.post("/evaluate-answers/")
async def evaluate_answers(payload: EvaluationRequest):
    questions = [q.dict() for q in payload.questions]
    answers = [a.dict() for a in payload.answers]
    """
    Evaluate student answers against the marking scheme.
    - questions: list of {number, question, max_marks, marking_scheme}
    - answers: list of {number, question, student_response}
    """

    # Merge by question number
    merged = []
    for q in questions:
        a = next((ans for ans in answers if str(ans["number"]) == str(q["number"])), None)
        if a:
            merged.append({
                "number": q["number"],
                "question": q["question"],
                "max_marks": q["max_marks"],
                "marking_scheme": q["marking_scheme"],
                "student_response": a["student_response"]
            })

    # Prepare prompt for Gemini
    prompt = (
        "You are an exam marker. For each question, use the marking scheme to score the student's response "
        "out of the max_marks, and provide a brief constructive comment. "
        "Return JSON only in this format:\n"
        "[{'number': <question_number>, 'score': <int>, 'comment': '<text>'}, ...]\n\n"
        f"Here is the data:\n{json.dumps(merged, indent=2)}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {
            "role": "systemInstruction",
            "parts": [{"text": "You are a strict but fair exam grader. Return JSON only, no explanations."}]
        },
        "generation_config": {
            "max_output_tokens": 2000,
            "temperature": 0.2,
            "topP": 1
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    clean_text = raw_text.strip("```json").strip("```").strip()

    return {"evaluation": json.loads(clean_text)}