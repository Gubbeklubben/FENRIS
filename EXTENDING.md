# Extending FENRIS

FENRIS is designed to be extended by plugins, primarily through the addition of new algorithms made up of a pair of `Synthesizer` and `Coordinator` components. The current version also supports adding new `Partitioner` and `Evaluator` implementations.

This page walks through the process of creating a new plugin called `myplugin` and adding skeleton implementations for three new `Synthesizer` components.

## Creating a plugin project

Navigate to your desired parent directory and run:

```bash
fenris plugin new myplugin
```

`myplugin` can be replaced with any name you prefer. This produces the following project structure:

```
myplugin/
├── pyproject.toml
└── myplugin/
    └── __init__.py
```

The only file with any content at this point is `pyproject.toml`:

```toml
# Created by fenris.cli.plugin.new using tomlkit 0.14.0.

[project]
name = "myplugin"
version = "0.1.0"
description = ""
authors = []
license = "MIT"

dependencies = ["fenris (>=0.1.0,<=0.1.0)"]
```

## Adding component scaffolds

To generate skeleton implementations for new synthesizers, run:

```bash
fenris plugin extend myplugin synthesizers synth1 synth2 synth3
```

The project now contains a few more files:

```
myplugin/
├── pyproject.toml
└── myplugin/
    ├── __init__.py
    └── synthesizers/
        ├── __init__.py
        ├── synth1.py
        ├── synth2.py
        └── synth3.py
```

Entry points for each synthesizer have been added to `pyproject.toml`:

```toml
# Created by fenris.cli.plugin.new using tomlkit 0.14.0.

[project]
name = "myplugin"
version = "0.1.0"
description = ""
authors = []
license = "MIT"

dependencies = ["fenris (>=0.1.0,<=0.1.0)"]

[project.entry-points."fenris.synthesizers"]
synth1 = "myplugin.synthesizers.synth1:Synth1"
synth2 = "myplugin.synthesizers.synth2:Synth2"
synth3 = "myplugin.synthesizers.synth3:Synth3"
```

By default, implementations are grouped per component type. This behavior can be suppressed by passing the `--no-group` flag. Each generated file contains a skeleton like the following:

```python
from typing import ClassVar

from pandas import DataFrame

from fenris.core.algorithm.synthesizer import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
    Synthesizer,
    TrainContext,
)
from fenris.core.payload import ArraysTarget, Payload


class Synth1(Synthesizer):
    SUPPORTED_COORDINATORS: ClassVar[set[str]]

    @property
    def arrays_target(self) -> ArraysTarget:
        raise NotImplementedError()

    def global_init(
        self,
        df: DataFrame,
        context: GlobalInitContext,
    ) -> GlobalInitArtifacts:
        raise NotImplementedError()

    def train(
        self,
        request: Payload,
        df: DataFrame,
        context: TrainContext,
    ) -> Payload:
        raise NotImplementedError()

    def sample(
        self,
        request: Payload,
        context: SampleContext,
    ) -> DataFrame:
        raise NotImplementedError()
```

## Installing and verifying

Install the plugin into the same environment where FENRIS is installed:

```bash
pip install ./myplugin
```

After installation, `fenris show synthesizers` should list the newly added implementations alongside the built-in ones. Running this command is a useful sanity check before wiring up a real implementation.

## Sharing plugins

Plugins can be shared via PyPI or any other package repository. Note that all plugins are treated as fully trusted source code &ndash; only install plugins from sources you trust.
