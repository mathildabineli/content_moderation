from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.inference import ClassifierService
from src.train_teacher import train_teacher_step
# Import interne (assure-toi que ces fichiers existent dans src/)


app = FastAPI()



@app.post("/train_teacher")
def train_teacher():
    

    try:
        labels, text_col = train_teacher_step()
        return {"status": "success", "labels": labels, "text_column": text_col}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PredictInput(BaseModel):
    texts: List[str]

@app.post("/predict")
def predict(input: PredictInput):

    try:
        service = ClassifierService(
            model_path="artifacts/teacher/best",
            tokenizer_path="artifacts/teacher/best",
            thresholds_path="artifacts/teacher_thresholds.json"
        )
        preds = service.predict(input.texts)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
