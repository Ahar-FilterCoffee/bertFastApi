from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer
import tensorflow as tf

from datetime import datetime
# Initialize FastAPI
app = FastAPI()

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = tf.saved_model.load("/home/aditya/Projects/Ahar-FilterCoffee/BertModel")

# Define the input data model
class TextData(BaseModel):
    text: str

# Define the prediction endpoint
@app.post("/predict/")
async def predict(data: TextData):
    try:
        preData={
        '5 January 2024': 'Saturday+Makar Sankranti or Pongal',
        '26 January 2024': 'Friday+Thaipusam',
        '14 February 2024': 'Wednesday+Vasant Panchami',
        '8 March 2024': 'Friday+Maha Shivaratri',
        '20 March 2024': 'Wednesday+Hindi New Year',
        '24 March 2024': 'Sunday+Holika Dahan',
        '25 March 2024': 'Monday+Holi',
        '9 April 2024': 'Tuesday+Ugadi or Gudi Padwa or Telugu New Year',
        '13 April 2024': 'Saturday+Vaisakhi or Baisakhi or Vishu',
        '14 April 2024': 'Sunday+Tamil New Year',
        '15 April 2024': 'Monday+Bengali New Year or Bihu',
        '17 April 2024': 'Wednesday+Ramanavami',
        '23 April 2024': 'Tuesday+Hanuman Jayanti',
        '10 April 2024': 'Friday+Akshaya Tritiya',
        '6 April 2024': 'Thursday+Savitri Pooja',
        '7 July 2024': 'Sunday+Puri Rath Yatra',
        '21 July 2024': 'Sunday+Guru Purnima',
        '9 August 2024': 'Friday+Nag Panchami',
        '16 August 2024': 'Friday+Varalakshmi Vratam',
        '19 August 2024': 'Monday+Raksha Bandhan',
        '26 August 2024': 'Monday+Krishna Janmashtami',
        '7 September 2024': 'Saturday+Ganesh Chaturthi',
        '16 September 2024': 'Monday+Vishwakarma Puja',
        '2 October 2024': 'Wednesday+Mahalaya Amavasya',
        '3 October 2024': 'Thursday+Navaratri begins',
        '11 October 2024': 'Friday+Navaratri ends or Maha Navami',
        '12 October 2024': 'Saturday+Dusshera',
        '16 October 2024': 'Wednesday+Sharad Purnima',
        '20 October 2024': 'Sunday+Karwa Chauthi',
        '29 October 2024': 'Tuesday+Dhan Teras',
        '1 November 2024': 'Friday+Diwali',
        '3 November 2024': 'Sunday+Bhai Dooj',
        '7 November 2024': 'Thursday+Chhath Puja',
        '15 November 2024': 'Friday+Kartik Poornima',
        '11 December 2024': 'Wednesday+Geeta Jayanti',
        '15 December 2024': 'Sunday+Dhanu Sankranti'}
        current_date = datetime.now()

# Format the date
        formatted_date = current_date.strftime("%d %B %Y")
        print(formatted_date)
        txt=[]
        if formatted_date in preData:
            txt.append(preData[formatted_date]+"restaurant")
            txt.append(preData[formatted_date]+"temple")
        else:
            day=current_date.strftime("%A")
            txt.append(formatted_date+day+"restaurant")
            txt.append(formatted_date+day+"temple")


        # Tokenize the input text
        #[data.text]
        encoding = tokenizer(txt, max_length=23, padding='max_length', truncation=True, return_tensors='tf')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Make predictions
        predictions = model([input_ids, attention_mask])
        print(predictions.numpy())
        # Output the predictions
        return {"prediction": [ float(predictions.numpy()[i]) for i in range(len(predictions.numpy()))]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the app, use the command:
# uvicorn script_name:app --reload
