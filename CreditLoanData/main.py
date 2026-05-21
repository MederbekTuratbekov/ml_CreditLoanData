import uvicorn
import joblib
from pydantic import BaseModel
from fastapi import FastAPI
from pyexpat import features
from pathlib import Path



BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / 'model_CreditLoanData.pkl')
scaler = joblib.load(BASE_DIR / 'scaler_CreditLoanData.pkl')

app = FastAPI()

in_home = ['OTHER', 'OWN', 'RENT']

class PersonSchema(BaseModel):
    person_age: int
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: int
    loan_int_rate: float
    credit_score: int

    class Config:
        from_attributes = True

@app.post('/predict/')
async def predict(person: PersonSchema):
    person_dict = dict(person)

    new_home = person_dict.pop('person_home_ownership')
    home1_0 = [1 if new_home == i else 0 for i in in_home]

    new_data = list(person_dict.values()) + home1_0
    scaled_date = scaler.transform([new_data])
    prediction = model.predict(scaled_date)[0]
    proba = model.predict_proba(scaled_date)[0][1]

    the_banks_response = 'Банк одобрил выдачу кредита' if prediction == 1 else 'Банк отклонил выдачу кредита'
    probability_of_approval = f'Вероятность одобрения банка: {round(proba * 100, 2)}%'
    return {
        'loan_status': the_banks_response,
        'probability_of_approval': probability_of_approval
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
