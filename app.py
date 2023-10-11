import socketio
import base64
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import io

# Load model and data
model = CNN.CNN(39)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Create a new Socket.IO server
sio = socketio.Server(cors_allowed_origins="*")
app = socketio.WSGIApp(sio)

# Event handler for the 'message' event
@sio.event
def image(sid, data):
    print('Received message:', data)

    try:
        image_bytes = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_bytes))

        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))
        output = model(input_data)
        output = output.detach().numpy()
        index = np.argmax(output)

        print(index)
        response_data = str(index)
  
        sio.emit('prediction_results', response_data)
    except Exception as e:
        print('Error processing image:', e)
        sio.emit('prediction_results', {'error': 'Error processing image'}, room=sid)

# Run the server
if __name__ == '__main__':
    import eventlet
    eventlet.wsgi.server(eventlet.listen(('localhost', 5000)), app)
