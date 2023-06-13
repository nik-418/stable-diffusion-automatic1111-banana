# import torch
import modules.safe as safe
import webui

# torch.load = safe.unsafe_torch_load
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def noop(*args, **kwargs):
    pass

# def register_model():
#     global model
#     try:
#         from modules import shared, sd_hijack
#         shared.sd_model = model
#         sd_hijack.model_hijack.hijack(model)
#     except:
#         print("Failed to hijack model.")

# def init():
#     global model
#     import modules.sd_models
#     modules.sd_models.list_models()
#     modules.sd_models.list_models = noop
#     model = modules.sd_models.load_model()
#     modules.sd_models.load_model = noop
#     register_model()

from potassium import Potassium, Request, Response

# from transformers import pipeline
import torch

from fastapi.testclient import TestClient

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    torch.load = safe.unsafe_torch_load
    device = 0 if torch.cuda.is_available() else -1 #

    import modules.sd_models
    modules.sd_models.list_models()
    modules.sd_models.list_models = noop
    model = modules.sd_models.load_model()
    modules.sd_models.load_model = noop

    try:
        from modules import shared, sd_hijack
        shared.sd_model = model
        sd_hijack.model_hijack.hijack(model)
    except:
        print("Failed to hijack model.")
    
    # model = pipeline('fill-mask', model='bert-base-uncased', device=device)
   
    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    # webui.api_only()

    # def inference(request: Request):
    # global client
    # body = await request.body()
    # model_input = json.loads(body)
    model_input = request.json
    
    params = None
    mode = 'default'

    if 'endpoint' in model_input:
        endpoint = model_input['endpoint']
        if 'params' in model_input:
            params = model_input['params']
    else:
        mode = 'banana_compat'
        endpoint = 'txt2img'
        params = model_input

    if endpoint == 'txt2img':
        if 'num_inference_steps' in params:
            params['steps'] = params['num_inference_steps']
            del params['num_inference_steps']
        if 'guidance_scale' in params:
            params['cfg_scale'] = params['guidance_scale']
            del params['guidance_scale']

    client = TestClient(app)

    if params is not None:
        response = client.post('/sdapi/v1/' + endpoint, json = params)
    else:
        response = client.get('/sdapi/v1/' + endpoint)

    output = response.json()

    if mode == 'banana_compat' and 'images' in output:
        output = {
            "base64_output": output["images"][0]
        }   

    return output

    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model(prompt)

    return Response(
        json = {"outputs": outputs[0]}, 
        status=200
    )

if __name__ == "__main__":
    webui.api_only()
    app.serve()
