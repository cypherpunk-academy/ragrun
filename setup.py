from setuptools import setup, find_packages

setup(
    name="ragrun",
    version="0.2.1",
    description="RAG runner, configured by ragprep.",
    author="",
    author_email="",
    license="MIT",
    packages=["assistants", "app", "assistants.config", "assistants.templates"],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.95.0",
        "pydantic>=1.10.7",
        "httpx>=0.24.0",
        "python-dotenv>=1.0.0",
        "jinja2>=3.1.2",
        "pinecone-client>=2.2.2",
        "python-multipart>=0.0.6",
        "uvicorn>=0.22.0"
    ],
    package_data={
        "assistants": ["templates/*.mdt", "config/*.json"],
    },
) 