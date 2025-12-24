import langchain
print(dir(langchain))
try:
    import langchain.chains
    print("langchain.chains imported")
    print(dir(langchain.chains))
except ImportError as e:
    print(f"Error importing langchain.chains: {e}")
