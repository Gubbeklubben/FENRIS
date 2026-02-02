from flwr.clientapp import ClientApp
from flwr.common import Message, Context


app = ClientApp()


@app.train()
def train(message: Message, context: Context) -> Message:
    # Load data
    # Call synthesizer factory
    # Set synthesizer weights from converted message content
    # Call synthesizer.train
    # Get synthesizer weights / other relevant stuff
    # Convert to Flower Message and return it
    pass


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    pass
