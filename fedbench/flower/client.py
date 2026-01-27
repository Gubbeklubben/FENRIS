from flwr.clientapp import ClientApp
from flwr.common import Message, Context


app = ClientApp()


@app.train()
def train(message: Message, context: Context) -> Message:
    pass


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    pass
