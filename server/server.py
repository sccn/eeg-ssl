# server.py
import sys
sys.path.append('../')
import litserve as ls
from libs.ssl_model import LitSSL, VGGSSL
import torch

# (STEP 1) - DEFINE THE API (compound AI system)
class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        # setup is called once at startup. Build a compound AI system (1+ models), connect DBs, load data, etc...
        self.model = LitSSL(VGGSSL()) #.load_from_checkpoint('path_to_your_model.ckpt')

    def decode_request(self, request):
        '''
        The decode_request method maps the network request into something your model can consume. 
        This can be images, text, or whatever was sent to your server.
        '''
        # assuming request is a json object with a key "data"
        data = request["data"]
        # convert data to torch tensor
        x = torch.tensor(data)
        assert len(x.shape) == 3, "x must be a 3D tensor (batch, channels, time)"
        return x

    def predict(self, x):
        '''
        The predict method is called with the output of decode_request. 
        We've split these up so that you can enable batching for more advanced use cases.
        '''
        embed = self.model.embed(x)
        return {"embedding": embed}

    def encode_response(self, output):
        '''
        The results of predict are passed to the encode_response method. 
        Use this to structure the response your server will return.
        '''
        data = output['embedding'].detach().cpu().numpy().tolist()
        json_encoders = {
            "output": data
        }
        return json_encoders

# (STEP 2) - START THE SERVER
if __name__ == "__main__":
    # scale with advanced features (batching, GPUs, etc...)
    server = ls.LitServer(SimpleLitAPI(), accelerator="auto", max_batch_size=1)
    server.run(port=8000)