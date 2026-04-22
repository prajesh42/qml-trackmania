# qml-trackmania

## create virtual env
python .venv tmrl_env
## activate the env
tmrl_env/Scripts/Activate
## install conda with version pywin32
conda install pywin32
## install tmrl
pip install tmrl
##  validate tmrl installation
python -m tmrl --install
## copy the config
1. Copy the quantum config from config_setup directory
2. Paste the content of quantum_config.json inside config.json in the user directory as C:\Users\(username)\TmrlData\config\config.json