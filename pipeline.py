import re
from typing import List, Dict, Any

# ---------- 1. Parse Question Paper ----------
def parse_question_paper(text: str) -> List[Dict[str, Any]]:
    """
    Takes OCR text of a question paper and extracts questions with numbering.
    Assumes format like '1. Question text (max X marks)...'
    """
    questions = []
    # naive regex split on numbering (1., 2., 3., etc.)
    parts = re.split(r'(\d+\.)', text)
    current_number = None
    buffer = []

    for part in parts:
        if re.match(r'\d+\.', part):
            # save previous question
            if current_number and buffer:
                questions.append({
                    "number": current_number,
                    "question": " ".join(buffer).strip(),
                    "max_marks": 10,  # placeholder
                    "marking_scheme": "not specified"  # placeholder
                })
                buffer = []
            current_number = int(part.strip("."))
        else:
            buffer.append(part)

    # save last
    if current_number and buffer:
        questions.append({
            "number": current_number,
            "question": " ".join(buffer).strip(),
            "max_marks": 10,
            "marking_scheme": "not specified"
        })

    return questions

# ---------- 2a. Parse Q&A Format ----------
def parse_student_answers(text: str, questions: List[Dict[str, Any]], student_details: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse when answers are written under the question (Q&A sheet).
    """
    answers = []
    q_map = {q["number"]: q for q in questions}

    parts = re.split(r'(\d+\.)', text)
    current_number = None
    buffer = []

    for part in parts:
        if re.match(r'\d+\.', part):
            if current_number and buffer:
                q = q_map.get(current_number, {})
                answers.append({
                    **q,
                    "student_response": " ".join(buffer).strip()
                })
                buffer = []
            current_number = int(part.strip("."))
        else:
            buffer.append(part)

    if current_number and buffer:
        q = q_map.get(current_number, {})
        answers.append({
            **q,
            "student_response": " ".join(buffer).strip()
        })

    return {"student_details": student_details, "answer_sheet": answers}

# ---------- 2b. Parse Answer-Only Format ----------
def parse_answer_only(text: str, questions: List[Dict[str, Any]], student_details: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse when answers are written on a separate sheet (Answer-only).
    Assumes answers are numbered 1., 2., 3., etc. in sequence.
    """
    answers = []
    q_map = {q["number"]: q for q in questions}

    parts = re.split(r'(\d+\.)', text)
    current_number = None
    buffer = []

    for part in parts:
        if re.match(r'\d+\.', part):
            if current_number and buffer:
                q = q_map.get(current_number, {})
                answers.append({
                    **q,
                    "student_response": " ".join(buffer).strip()
                })
                buffer = []
            current_number = int(part.strip("."))
        else:
            buffer.append(part)

    if current_number and buffer:
        q = q_map.get(current_number, {})
        answers.append({
            **q,
            "student_response": " ".join(buffer).strip()
        })

    return {"student_details": student_details, "answer_sheet": answers}

# ---------- 3. Dummy LLM Scoring ----------
def call_llm_marker(number, question, max_marks, marking_scheme, response):
    """
    Simulate LLM scoring. In production: call GPT or fine-tuned model.
    """
    score = min(len(response.split()) // 5, max_marks)  # toy scoring
    comment = "Good attempt, but elaborate more." if score < max_marks else "Excellent and detailed."
    return {"number": number, "score": score, "comment": comment}

# ---------- 4. End-to-End Processing ----------
def process_exam(question_paper_text: str, student_answer_text: str, student_details: Dict[str, str]) -> Dict[str, Any]:
    # 1. Parse questions
    questions = parse_question_paper(question_paper_text)

    # 2. Decide format
    if any(q in student_answer_text for q in ["1.", "2."]):
        student_report = parse_student_answers(student_answer_text, questions, student_details)
    else:
        student_report = parse_answer_only(student_answer_text, questions, student_details)

    # 3. Score responses
    scores = []
    for q in student_report["answer_sheet"]:
        result = call_llm_marker(q["number"], q["question"], q["max_marks"], q["marking_scheme"], q["student_response"])
        scores.append(result)

    # 4. Merge
    for q in student_report["answer_sheet"]:
        match = next((s for s in scores if s["number"] == q["number"]), None)
        if match:
            q["score"] = match["score"]
            q["comment"] = match["comment"]

    return student_report

# ---------- Example Run ----------
if __name__ == "__main__":
    question_paper_text = """
    1. Discuss 5 effects of world war II in the developing world. (10 marks)
    2. Define photosynthesis. (5 marks)
    3. Explain the process of respiration. (10 marks)
    """

    student_answer_text = """
    1. World war II led to economic decline, destruction of infrastructure, loss of human capital,
    political instability, and rise of independence movements.
    2. Photosynthesis is the process by which plants make food using sunlight.
    3. Respiration is the process of breaking down glucose to release energy in the presence of oxygen.
    """

    student_details = {"name": "Alex Munene", "student_id": 20995, "class": "Grade 12 Lenana"}

    report = process_exam(question_paper_text, student_answer_text, student_details)
    import json
    print(json.dumps(report, indent=2))
