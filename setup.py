from distutils.core import setup

with open('requirements.txt') as f:
    req = [i[:i.find('>')] + '(' + i[i.find('>'):] + ')' for i in f.read().splitlines()]

setup(
    name="pycoin",
    version="0.1.0",
    description="Algorithmic trading tools for cryptocurrencies",
    author="Djordje Pepic",
    author_email="djordje.m.pepic@gmail.com",
    packages=["pycoin", "pycoin.envs"],
    requires=req
)
