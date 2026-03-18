import pandas as pd

def load():
    """
    Load NSL-KDD dataset.
    Using KDDTrain+.txt for training data.
    """
    # Assuming the file has headers; if not, add column names
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    data = pd.read_csv('datasets/NSL_KDD/KDDTrain+.txt', header=None, names=columns)
    
    # AXAI-IDS requires binary classification (0 = Normal, 1 = Attack)
    if 'label' in data.columns:
        data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
    # AXAI-IDS Prevention: Drop data leak columns ('difficulty')
    if 'difficulty' in data.columns:
        data.drop(columns=['difficulty'], inplace=True)
        
    return data