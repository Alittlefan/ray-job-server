import yaml

def load_config(config_path):
    """
    从YAML读取环境配置
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    #特定字段转换成元组
    list_to_tuple = [
        'red_bigua_pos',
        'red_bazhua_pos',
        'blue_deploy_pos',
        'blue_target_pos'
    ]
    for field in list_to_tuple:
        if field in config:
            config[field] = [tuple(item) for item in config[field]]
    
    return config