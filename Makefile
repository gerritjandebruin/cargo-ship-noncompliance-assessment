PHONY: clean
clean:
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.ipynb_checkpoints' -exec rm -fr {} +
    
PHONY: install
install:
	git clone https://github.com/franktakes/teexgraph.git
	cd teexgraph/ && git reset --hard 0c4ebef4ee938aa842bf40d1aec8a66d95fd8a82 && make listener