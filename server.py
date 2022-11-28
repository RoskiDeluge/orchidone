import os
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
import replicate
import openai

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
def index():
  response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Prompt: A surreal image of an orchid floating in outer space in the style of Rene Magritte and Georgia O'Keffe\n\nPrompt: A hyperrealistic photo of a burning orchid in the style of Frida Kahlo and Ansel Adams\n\nPrompt: An abstract image of an orchid explosion in a futuristic landscape in the style of Vincent Van gough and Takashi Murakami \n\nPrompt:",
  temperature=0.9,
  max_tokens=140,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
  )
  imgprompt = response.choices[0].text
  model = replicate.models.get("stability-ai/stable-diffusion")
  sdorchid = model.predict(prompt=imgprompt)
  sdorchidimg = sdorchid[0]
  oaiorchid = openai.Image.create(prompt=imgprompt, n=1, size="512x512")
  oaiorchidimg = oaiorchid.data[0].url
  
  return render_template("index.html", sdorchidimg=sdorchidimg, oaiorchidimg=oaiorchidimg, imgprompt=imgprompt)

if __name__ == "__main__":
  app.run()
