import typer

from fedbench._plugins import algorithms, load_algorithm

app = typer.Typer()


@app.command()
def new(name: str):
    pass


@app.command()
def list_algorithms() -> None:
    print(*algorithms())


@app.command()
def run(
        algorithm_name: str,
        num_nodes: int = typer.Option(default=3)) -> None:

    alg = load_algorithm(algorithm_name)
    print(alg)
    #run_simulation(server_app, client_app, 10)


if __name__ == "__main__":
    app()
