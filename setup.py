from setuptools import setup, find_packages

setup(
    name='rag_chat',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'langchain-community==0.3.16',
        'langchain-text-splitters==0.3.5',
        'langchain-openai==0.3.4rc1',
        'langchain-core==0.3.34rc2',
        'langgraph==0.2.69',
    ],
)
