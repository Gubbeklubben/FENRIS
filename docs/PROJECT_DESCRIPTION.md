# PROJECT DESCRIPTION

## Project Title

An Extensible Benchmarking Framework for Federated Synthetic Tabular Data Generators

## Background and Motivation

The generation of synthetic tabular data has emerged as an important approach for enabling
privacy-preserving data sharing and supporting downstream machine learning in sensitive
domains such as healthcare and finance. Centralized synthetic data generators such as
CTGAN and TVAE have been widely studied, yet their reliance on pooled raw data limits
applicability in settings constrained by data sovereignty and regulatory requirements.
Federated synthetic data generators provide a promising alternative by allowing collaborative
model training across multiple decentralized data silos without direct data exchange.

## Project Objectives

This thesis focuses on developing an extensible benchmarking framework for federated
synthetic tabular data generators. To demonstrate the usefulness and capabilities of the
benchmarking tool, three state-of-the-art federated synthetic data generation algorithms will
be implemented and systematically compared:

1. **Fed-TGAN:** A tabular GAN adapted to horizontal federated learning
2. **VT-GAN:** A vertical federated GAN with integrated **differential** privacy mechanisms
3. **FedTabDiff:** A federated **diffusion** model designed for mixed-type tabular data

These implementations serve dual purposes: they validate the framework's functionality and
provide concrete examples of how researchers can integrate new algorithms. The
comparative evaluation demonstrates the benchmarking tool's ability to assess federated
synthetic data generators across multiple dimensions.

## Datasets

The project will use publicly available datasets including:
● UCI Heart Disease Dataset
● Breast Cancer Wisconsin Dataset
● NCCTG Lung Cancer Dataset

## Evaluation Dimensions

The models will be evaluated across four dimensions:

1. **Fidelity and Utility:** Measured by statistical similarity, correlation preservation, and
    downstream machine learning performance in train-on-synthetic-test-on-real tasks
2. **Privacy:** Assessed through membership inference and attribute inference attacks
    and, where available, **differential** privacy accounting
3. **Fairness:** Examined through group fairness measures including demographic parity
    and equalized odds
4. **Scalability:** Analyzed through training **efficiency** , communication overhead, and
    computational footprint in federated environments

## Technical Implementation

The core contribution is the development of an extensible benchmarking framework built on
the Flower federated learning platform using Python. The framework will be designed with
the following architectural principles:

### 1. Modular Architecture

○ Clear separation between data loading, model implementation, evaluation
metrics, and orchestration
○ Abstract base classes defining interfaces for federated synthetic data
generators
○ Plugin-style system allowing new algorithms to be added with minimal code
changes

### 2. Standardized Evaluation Pipeline

○ Unified evaluation interface for all models
○ Consistent metric computation across different generator types
○ Reusable evaluation modules for fidelity, privacy, fairness, and scalability

### 3. Documentation for Extension

○ Comprehensive documentation explaining the framework architecture
○ Step-by-step guide for implementing new federated synthetic data generators
○ Code examples demonstrating how to extend the base classes
○ Tutorial showing the integration process from algorithm implementation to benchmark inclusion.

The initial implementation will integrate Fed-TGAN, VT-GAN, and FedTabDiff as
demonstration cases, with clear documentation showing how these integrations serve as
templates for future algorithm additions.

## Expected Contributions

The contributions of this work are fourfold:

1. **An extensible benchmarking framework:** A modular, well-documented framework
    for evaluating federated synthetic data generators, designed to accommodate future
    algorithms with minimal modification
2. **Demonstration through comparative analysis:** Implementation and systematic
    comparison of three diverse federated approaches (Fed-TGAN, VT-GAN, FedTabDiff)
    that validate the benchmarking tool's functionality and demonstrate its evaluation
    capabilities across different architectural patterns
3. **Comprehensive evaluation methodology:** An empirical assessment of the
    trade-offs between the three implemented approaches across fidelity, privacy,
    fairness, and scalability dimensions, showcasing the benchmarking tool's analytical
    capabilities
4. **Extension guidelines and reference implementations:** Practical documentation,
    tutorials, and working code examples enabling researchers to integrate new federated
    synthetic data generation algorithms into the benchmarking framework
By implementing and comparing GAN- and **diffusion** -based federated approaches within an
extensible framework, this thesis delivers the first unified, reusable evaluation platform for
federated synthetic tabular data generators and contributes to the advancement of
privacy-aware data sharing solutions.

## Key References and Resources

● Federated Learning Framework:
○ Flower: <https://flower.ai/>
● Federated Synthetic Data Generators:
○ Fed-TGAN: <https://github.com/zhao-zilong/Fed-TGAN>
○ GTV (VT-GAN): <https://github.com/zhao-zilong/gtv>
○ FedTabDiff: <https://github.com/sattarov/FedTabDih>

## Project Timeline

● Project Duration: January 2026 - May 2026
● Scope Flexibility: The project scope may be adjusted based on progress and mutual
agreement between OsloMet and the Partner. Any scope modifications should be
discussed during the bi-weekly meetings and documented in writing.

**Month Milestones

### JANUARY 2026 Project Setup and Literature Review**

* Initial meeting with supervisors from OsloMet and Partner
* Literature review on federated learning and synthetic data
generation
* Review existing implementations of Fed-TGAN, VT-GAN, and
FedTabDiff
* Design framework architecture with focus on extensibility
* Define abstract base classes and interfaces for the
benchmarking tool
* Environment setup: Install Flower framework and dependencies
* Acquire and prepare datasets (UCI Heart Disease, Breast
Cancer Wisconsin, NCCTG Lung Cancer)
* Define evaluation metrics and experimental design

### FEBRUARY 2026 Framework Development and First Implementation

* Develop core framework components (data loaders, evaluation
pipeline, orchestration)
* Implement abstract base classes for federated synthetic data
generators
* Integrate Fed-TGAN as first reference implementation
* Set up standardized evaluation interface
* Begin documentation of framework architecture

### MARCH 2026 Additional Implementations and Testing

* Integrate VT-GAN into the framework
* Integrate FedTabDiff into the framework
* Refine framework interfaces based on implementation
experience
* Conduct preliminary experiments to validate framework
functionality
* Document integration process and design patterns

### APRIL 2026 Comprehensive Evaluation

* Complete fidelity and utility evaluations (statistical similarity,
correlation preservation, ML performance)
* Conduct privacy assessments (membership inference, attribute
inference attacks)
* Perform fairness analysis (demographic parity, equalized odds)
* Analyze scalability metrics (training efficiency , communication
overhead, computational footprint)
* Compare results across all three models and evaluation
dimensions

### MAY 2026 Documentation, Thesis Writing, and Finalization

* Complete developer documentation and extension guidelines
* Write step-by-step tutorial for adding new algorithms
* Prepare code examples demonstrating framework extension
* Write thesis chapters covering framework design, methodology,
results, and analysis
* Prepare visualizations and tables for results
* Code documentation, testing, and repository cleanup
* Final thesis review and revisions
* Deadline: Thesis submission by [specific date to be determined]

## Expected Deliverables

1. **Extensible benchmarking framework:** Modular codebase with clear architecture for
    integrating new algorithms
2. **Three reference implementations:** Fed-TGAN, VT-GAN, and FedTabDiff integrated
    into the framework
3. **Comprehensive evaluation results:** Comparative analysis across fidelity, privacy,
    fairness, and scalability dimensions
4. **Developer documentation:**
    ○ Architecture overview and design patterns
    ○ API documentation for base classes and interfaces
    ○ Step-by-step guide for implementing new algorithms
    ○ Integration tutorial with code examples
5. **Bachelor thesis:** Complete documentation of framework design, methodology,
    evaluation results, and analysis
6. **Source code repository:** Well-documented, tested code with README and
    contribution guidelines
