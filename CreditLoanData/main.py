from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / 'model_CreditLoanData.pkl')
scaler = joblib.load(BASE_DIR / 'scaler_CreditLoanData.pkl')

app = FastAPI()

IN_HOME = ['OTHER', 'OWN', 'RENT']


class PersonSchema(BaseModel):
    person_age: int
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: int
    loan_int_rate: float
    credit_score: int


@app.post('/predict/')
async def predict(person: PersonSchema):
    home_encoded = [1 if person.person_home_ownership == home_type else 0 for home_type in IN_HOME]

    features_list = [
                        person.person_age,
                        person.person_income,
                        person.person_emp_exp,
                        person.loan_amnt,
                        person.loan_int_rate,
                        person.credit_score
                    ] + home_encoded

    scaled_data = scaler.transform([features_list])
    prediction = model.predict(scaled_data)[0]
    proba = model.predict_proba(scaled_data)[0][1]

    the_banks_response = 'Банк одобрил выдачу кредита' if prediction == 0 else 'Банк отклонил выдачу кредита'
    proba_approval = model.predict_proba(scaled_data)[0][0]
    probability_of_approval = f'Вероятность одобрения банка: {round(proba_approval * 100, 2)}%'

    return {
        'loan_status': the_banks_response,
        'probability_of_approval': probability_of_approval
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
