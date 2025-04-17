from pathlib import Path
import socket
def get_machine_info():
    try:
        # Obtenir le nom d'hôte de la machine
        hostname = socket.gethostname()
        print(f"Nom de l'hôte : {hostname}")

        # Obtenir l'adresse IP locale
        local_ip = socket.gethostbyname(hostname)
        print(f"Adresse IP locale : {local_ip}")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

    return hostname

is_pc = get_machine_info() == 'caron-Precision-7780'

# Arboressence
if not is_pc:
    rootDisk = Path('/Work/Users/ncaron')
    root = rootDisk
    root_graph = root / 'GNN'
    root_target = root / 'Target'

else:

    rootDisk = Path('/media/caron/X9 Pro/travaille/Thèse/')
    root = rootDisk 
    root_graph = root / 'Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN'
    root_target = root / 'Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/Target'