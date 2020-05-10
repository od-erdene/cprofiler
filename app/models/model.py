from models.nearul_net import Network


class Model2(object):

    def __init__(self):
        self._model = Network()
        self._model.load_state_dict(torch.load("./params/best_model.ckpt"))

    def predict(self, csv_file):

        return "test"
        
