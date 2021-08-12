# Tensorboard

Jean Zay has a specific procedure to check tensorboard logs detailed [here](http://www.idris.fr/eng/jean-zay/pre-post/jean-zay-jupyter-notebook-eng.html). It essentially boils down to:
```bash
module load tensorflow-gpu/py3/2.3.0 # You can use your own env or other JZ existing envs
jupyter tensorboard enable --user
idrjup
```
Please note that you need to connect from the declared IP adress.

# Potential errors

On Jupyter, if you run into an *Invalid credentials* error, or a *Jupyter tensorboard extension error*, as suggested by Rémi Lacroix, you can remove the `~/.jupyter` folder (command: `rm -rf ~/.jupyter`) and restart the procedure from scratch. In particular, make sure you re-activate the tensorboard plugin for your user: `jupyter tensorboard enable --user`. It generally fixes that kind of problems.
