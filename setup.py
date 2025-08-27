from setuptools import setup, find_packages

setup(
    name="vllm-framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.21.0",
        "triton>=2.0.0",
        "sentencepiece>=0.1.99",
        "accelerate>=0.20.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
    ],
    author="benyuereal",
    author_email="18850575164@163.com",
    description="A high-throughput and memory-efficient inference engine for LLMs",
    keywords="llm inference serving transformer deeplearning",
    python_requires=">=3.8",
)