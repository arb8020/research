import gradio as gr
import torch
from huggingface_hub import hf_hub_download

def load_model_from_hub(repo_id, filename):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = torch.load(model_path, weights_only=False, map_location='cpu')
    model.eval()
    return model

def predict(text, model):
    with torch.no_grad():
        output = model(text)
        return float(output)

def create_gradio_app():
    repo_id = "jane-street/2025-03-10"
    model_filename = "model_3_11.pt"
    model = load_model_from_hub(repo_id, model_filename)
    
    with gr.Blocks() as demo:
        gr.Markdown('''        
Today I went on a hike and found a pile of tensors hidden underneath a neolithic burial mound!

I sent it over to the local neural plumber, and they managed to cobble together this.

**[model.pt](https://huggingface.co/jane-street/2025-03-10/tree/main)**
            
Anyway, I'm not sure what it does yet, but it must have been important to this past civilization. 
Maybe start by looking at the last two layers. 
        ''')
        
        input_text = gr.Textbox(label="Model Input", value='vegetable dog') # two words?
        output = gr.Number(label="Model Output")
        
        input_text.submit(fn=lambda x: predict(x, model), inputs=input_text, outputs=output)
        
        gr.Markdown('''
If you do figure it out, please let us know at *archaeology@janestreet.com*.

---

Solved by
 - Noa Nabeshima and Collin Gray 
 - Andrew Peterson
 - Alex Waese-Perlman
 - David Rapisarda and Jayant Khatkar
 - Ryan Bruntz
 - Sam Corbett
 - Can Elbirlik
 - Benedict Davies
 - Вадим Калашников
 - Géza Herman
 - Jack Murphy
 - Snehal Verma
 - Sean Osier
 - Sam Patterson
 - Nikita Klyuchnikov
 - Xiao Zheng
 - Fedor Korotkiy
 - Nick Griffiths
 - Ujas Shah
 - Oleh Feilo
 - Павел Дергунов
 - Gregor Kikelj
 - Adir Kobovich
 - Logan Seaburg
 - Amber Lin
 - Sai Nane
 - Brice Marnier
 - Alan McCreanor
 - Han Li
 - Pablo Andres Focke
 - Andi Qu
 - Ryan Mathieu

---

*Learn more at [janestreet.com](https://jane-st.co/3YfP5WK)*.
        ''')

    demo.queue(max_size=1_000)
    
    return demo

    
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(show_api=False)
